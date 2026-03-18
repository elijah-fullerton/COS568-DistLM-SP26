# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa)."""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import time

import matplotlib
matplotlib.use("Agg")  # Headless-friendly plotting (for servers/cluster nodes)
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.profiler
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

# import a previous version of the HuggingFace Transformers package
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForSequenceClassification, BertTokenizer,
                                  RobertaConfig,
                                  RobertaForSequenceClassification,
                                  RobertaTokenizer,
                                  XLMConfig, XLMForSequenceClassification,
                                  XLMTokenizer, XLNetConfig,
                                  XLNetForSequenceClassification,
                                  XLNetTokenizer)

from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils_glue import (compute_metrics, convert_examples_to_features,
                        output_modes, processors)

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, RobertaConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)

def init_distributed_mode(args):
    """
    Initialize torch.distributed with TCP init_method.
    The assignment uses `--local_rank` as the rank (0..world_size-1).
    """
    if args.local_rank == -1:
        args.world_size = 1
        args.rank = 0
        return

    if args.world_size is None or args.world_size < 1:
        raise ValueError("In distributed mode you must pass a valid --world_size.")
    if not args.master_ip:
        raise ValueError("In distributed mode you must pass a valid --master_ip.")
    if args.master_port is None:
        raise ValueError("In distributed mode you must pass a valid --master_port.")

    backend = "nccl" if args.device.type == "cuda" else "gloo"
    if backend == "nccl":
        # Required by NCCL when using multiple GPU processes per node.
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)

    torch.distributed.init_process_group(
        backend=backend,
        init_method=f"tcp://{args.master_ip}:{args.master_port}",
        world_size=args.world_size,
        rank=args.local_rank,
    )
    args.rank = args.local_rank

def sync_gradients_gather_scatter(args, model):
    """
    Average gradients across ranks using gather/scatter in gloo or nccl.
    Rank 0 gathers all grads, computes the element-wise mean, and scatters it back.
    """
    if args.local_rank == -1 or args.world_size == 1:
        return

    params_with_grad = []
    grad_shapes = []
    grad_numels = []
    flat_grads = []
    for p in model.parameters():
        if p.grad is None:
            continue
        params_with_grad.append(p)
        grad_shapes.append(tuple(p.grad.data.shape))
        grad_numels.append(p.grad.data.numel())
        flat_grads.append(p.grad.data.contiguous().view(-1))

    if not flat_grads:
        return

    flat_grad = torch.cat(flat_grads)

    gather_list = None
    if args.local_rank == 0:
        gather_list = [torch.zeros_like(flat_grad) for _ in range(args.world_size)]
    torch.distributed.gather(flat_grad, gather_list=gather_list, dst=0)

    scatter_list = None
    if args.local_rank == 0:
        mean_flat = torch.zeros_like(flat_grad)
        for g in gather_list:
            mean_flat.add_(g)
        mean_flat.div_(args.world_size)
        scatter_list = [mean_flat.clone() for _ in range(args.world_size)]

    out_flat = torch.zeros_like(flat_grad)
    torch.distributed.scatter(out_flat, scatter_list=scatter_list, src=0)

    offset = 0
    for p, shape, numel in zip(params_with_grad, grad_shapes, grad_numels):
        new_grad = out_flat[offset:offset + numel].view(shape)
        p.grad.data.copy_(new_grad)
        offset += numel


def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    args.train_batch_size = args.per_device_train_batch_size
    is_distributed = args.local_rank != -1
    if is_distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=args.world_size,
            rank=args.local_rank,
            shuffle=True,
            seed=args.seed,
        )
    else:
        train_sampler = RandomSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per device = %d", args.per_device_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    opt_step_times = []
    step_history = []
    loss_history = []
    num_reported_minibatches = 0

    # Task 4 profiling: skip first opt step, profile next three.
    profiling_enabled = bool(getattr(args, "profile", False)) and (args.local_rank in [-1, 0])
    prof = None
    if profiling_enabled:
        os.makedirs(args.output_dir, exist_ok=True)
        rank_tag = f"rank{args.local_rank}" if args.local_rank != -1 else "single"
        trace_path = os.path.join(args.output_dir, f"trace_task4_task2a_gather_scatter_{rank_tag}.json")

        activities = [torch.profiler.ProfilerActivity.CPU]
        if args.device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        def _trace_handler(p):
            p.export_chrome_trace(trace_path)

        prof = torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(wait=1, warmup=0, active=3, repeat=1),
            on_trace_ready=_trace_handler,
            record_shapes=False,
            with_stack=False,
        )
        prof.__enter__()

    for epoch_idx in train_iterator:
        if is_distributed:
            train_sampler.set_epoch(epoch_idx)
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            if step % args.gradient_accumulation_steps == 0:
                iter_start = time.perf_counter()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'labels':         batch[3]}
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # Task 1 requirement: print loss for first five minibatches (single-node only).
            if (not is_distributed) and num_reported_minibatches < 5 and args.local_rank in [-1, 0]:
                print(f"Minibatch {num_reported_minibatches + 1} loss: {loss.item()}")
                num_reported_minibatches += 1

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if is_distributed:
                    sync_gradients_gather_scatter(args, model)
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                if is_distributed:
                    sync_gradients_gather_scatter(args, model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                iter_end = time.perf_counter()
                optimizer.step()
                scheduler.step() # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                step_history.append(global_step)
                loss_history.append(loss.item())

                if prof is not None:
                    prof.step()

                if is_distributed:
                    opt_step_times.append(iter_end - iter_start)
                    # Loss curve per node for Task 2(a) logging.
                    print(f"rank {args.local_rank} global_step {global_step} loss {loss.item()}")

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
        
        evaluate(args, model, tokenizer, prefix=f"epoch_{epoch_idx}")
        if is_distributed:
            torch.distributed.barrier()

    if prof is not None:
        prof.__exit__(None, None, None)

    if is_distributed and args.local_rank == 0 and len(opt_step_times) > 1:
        avg_time = sum(opt_step_times[1:]) / len(opt_step_times[1:])
        print(f"Task2(a) avg opt-step time excl first: {avg_time:.6f}s")

    # Save training loss curve (one curve per rank/node).
    if len(step_history) > 0:
        rank_tag = f"rank{args.local_rank}" if args.local_rank != -1 else "single"
        os.makedirs(args.output_dir, exist_ok=True)

        data_path = os.path.join(args.output_dir, f"loss_curve_{rank_tag}.txt")
        with open(data_path, "w") as f:
            for s, l in zip(step_history, loss_history):
                f.write(f"{s}\t{l}\n")

        plt.figure()
        plt.plot(step_history, loss_history)
        plt.xlabel("Optimization step")
        plt.ylabel("Loss")
        plt.title(f"Training loss ({rank_tag})")
        plot_path = os.path.join(args.output_dir, f"loss_curve_{rank_tag}.png")
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    # In distributed mode, all ranks must participate in evaluation to avoid
    # mismatched barrier calls inside `load_and_cache_examples()`.
    if args.local_rank != -1:
        torch.distributed.barrier()

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + '-MM') if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_device_eval_batch_size
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM and RoBERTa don't use segment_ids
                          'labels':         batch[3]}
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        # Only rank 0 writes evaluation outputs to disk.
        if args.local_rank in [-1, 0]:
            safe_prefix = prefix if prefix else "final"
            safe_prefix = safe_prefix.replace("/", "_").replace("\\", "_")
            output_eval_file = os.path.join(eval_output_dir, f"eval_results_{safe_prefix}.txt")
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} *****".format(prefix))
                writer.write(f"eval_loss = {eval_loss}\n")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))

    if args.local_rank != -1:
        torch.distributed.barrier()
    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ['mnli', 'mnli-mm'] and args.model_type in ['roberta']:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1] 
        examples = processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        features = convert_examples_to_features(examples, label_list, args.max_seq_length, tokenizer, output_mode,
            cls_token_at_end=bool(args.model_type in ['xlnet']),            # xlnet has a cls token at the end
            cls_token=tokenizer.cls_token,
            cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
            sep_token=tokenizer.sep_token,
            sep_token_extra=bool(args.model_type in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_device_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_device_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    # parser.add_argument("--eval_all_checkpoints", action='store_true',
    #                     help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank. If single-node training, local_rank defaults to -1.")
    parser.add_argument("--master_ip", type=str, default="",
                        help="For distributed training: IP address of the master node.")
    parser.add_argument("--master_port", type=int, default=None,
                        help="For distributed training: TCP port for init_process_group.")
    parser.add_argument("--world_size", type=int, default=None,
                        help="For distributed training: total number of processes/nodes.")
    parser.add_argument("--profile", action="store_true",
                        help="Task 4: enable torch.profiler to export a Chrome trace (rank 0 only).")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # set up (distributed) training
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, args.device, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Initialize distributed environment (if requested).
    init_distributed_mode(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)


    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation
    evaluate(args, model, tokenizer, prefix="")

if __name__ == "__main__":
    main()
