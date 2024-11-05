import logging
import os
import pickle
import random
import numpy as np

import torch

from torch.utils.data.distributed import DistributedSampler
import json
from sklearn.metrics import recall_score, precision_score, f1_score
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset

from utils.parserTool.utils import remove_comments_and_docstrings
import utils.parserTool.parse as ps
from utils.parserTool.parse import Lang
from .java_cfg import JAVA_CFG
from .python_cfg import PYTHON_CFG
from .php_cfg import PHP_CFG
from .c_cfg import C_CFG

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

logger = logging.getLogger(__name__)


def extract_pathtoken(source, path_sequence):
    seqtoken_out = []
    for path in path_sequence:
        seq_code = ''
        for line in path:
            if (line in source):
                seq_code += source[line]
        seqtoken_out.append(seq_code)
        if len(seqtoken_out) > 10:
            break
    if len(path_sequence) == 0:
        seq_code = ''
        for i in source:
            seq_code += source[i]
        seqtoken_out.append(seq_code)
    seqtoken_out = sorted(seqtoken_out, key=lambda i: len(i), reverse=False)
    return seqtoken_out


def extract_pathtoken(source, path_sequence):
    seqtoken_out = []
    for path in path_sequence:
        seq_code = ''
        for line in path:
            if line != 'exit' and (line in source):
                seq_code += source[line]
        seqtoken_out.append(seq_code)
        if len(seqtoken_out) > 5:
            break
    if len(path_sequence) == 0:
        seq_code = ''
        for i in source:
            seq_code += source[i]
        seqtoken_out.append(seq_code)
    seqtoken_out = sorted(seqtoken_out, key=lambda i: len(i), reverse=False)
    return seqtoken_out

class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 path_source,
                 idx,
                 label,
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.path_source = path_source
        self.idx = str(idx)
        self.label = label


def convert_examples_to_features(js, tokenizer, path_dict, args,language=None):
    clean_code, code_dict = remove_comments_and_docstrings(js['func'], language)
    code = ' '.join(clean_code.split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length

    if js['idx'] in path_dict:
        path_tokens1, cfg_allpath = path_dict[js['idx']]
    else:
        clean_code, code_dict = remove_comments_and_docstrings(js['func'], language)
        if language == "java":
            g = JAVA_CFG()
            code_ast = ps.tree_sitter_ast(clean_code, Lang.JAVA)
        elif language == "python":
            g = PYTHON_CFG()
            code_ast = ps.tree_sitter_ast(clean_code, Lang.PYTHON)
        elif language == "php":
            g = PHP_CFG()
            code_ast = ps.tree_sitter_ast(clean_code, Lang.PHP)
        elif language == "c":
            g = C_CFG()
            code_ast = ps.tree_sitter_ast(clean_code, Lang.C)

        s_ast = g.parse_ast_file(code_ast.root_node)
        num_path, cfg_allpath, _, _ = g.get_allpath()
        path_tokens1 = extract_pathtoken(code_dict, cfg_allpath)

    all_seq_ids = []
    for seq in path_tokens1:
        seq_tokens = tokenizer.tokenize(seq)[:args.block_size - 2]
        seq_tokens = [tokenizer.cls_token] + seq_tokens + [tokenizer.sep_token]
        seq_ids = tokenizer.convert_tokens_to_ids(seq_tokens)
        padding_length = args.block_size - len(seq_ids)
        seq_ids += [tokenizer.pad_token_id] * padding_length
        all_seq_ids.append(seq_ids)

    if len(all_seq_ids) < args.filter_size:
        for i in range(args.filter_size - len(all_seq_ids)):
            all_seq_ids.append(source_ids)
    else:
        all_seq_ids = all_seq_ids[:args.filter_size]
    return InputFeatures(source_tokens, source_ids, all_seq_ids, js['idx'], js['target'])


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None,pkl_file_path=None,language=None):
        self.examples = []
        if pkl_file_path:
            pkl_file = open(pkl_file_path,"rb")
        path_dict = pickle.load(pkl_file)

        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip()) 
                self.examples.append(convert_examples_to_features(js, tokenizer, path_dict, args,language))

        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info("input_tokens: {}".format([x.replace('\u0120', '_') for x in example.input_tokens]))
                logger.info("input_ids: {}".format(' '.join(map(str, example.input_ids))))
        pkl_file.close()

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label), torch.tensor(
            self.examples[i].path_source)
    

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, train_dataloader, train_dataset, eval_dataloader, eval_dataset, model, tokenizer,pkl_file_path=None,language=None):
    args.max_steps = args.epoch * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    args.num_train_epochs = args.epoch
    model.to(args.device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.max_steps * 0.1,
                                                num_training_steps=args.max_steps)


    if args.use_adapters:
        if language == "c":
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=args.c_max_steps * 0.1,
                                                            num_training_steps=args.c_max_steps)
        elif language == "java":
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=args.java_max_steps * 0.1,
                                                            num_training_steps=args.java_max_steps)
        elif language == "python":
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=args.python_max_steps * 0.1,
                                                            num_training_steps=args.python_max_steps)
        elif language == "php":
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                            num_warmup_steps=args.php_max_steps * 0.1,
                                                            num_training_steps=args.php_max_steps)        


    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_mrr = 0.0
    best_acc = 0.0
    best_f1 = 0.0
    early_stop = 0

    model.zero_grad()  
    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        early_stop += 1

        bar = tqdm(train_dataloader, total=len(train_dataloader))

        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            seq_inputs = batch[2].to(args.device)
            model.train()
            cross_entropy_loss, logits = model(seq_ids=seq_inputs, input_ids=inputs,
                                                                       labels=labels, language=language)                                                    
            loss = cross_entropy_loss



            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:

                    if args.local_rank == -1 and args.evaluate_during_training: 
                        results = evaluate(args, eval_dataloader, eval_dataset,  model, tokenizer, eval_when_training=True,pkl_file_path=pkl_file_path,language=language)
                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value, 4))

                    if results['eval_f1'] > best_f1:
                        best_acc = results['eval_acc']
                        best_f1 = results['eval_f1']
                        logger.info("  " + "*" * 20)
                        logger.info("  Best f1:%s", round(best_f1, 4))
                        logger.info("  " + "*" * 20)

                        model.save(idx,optimizer=None,model_dir='./checkpoints',mode=1)
                        early_stop = 0

def evaluate(args, eval_dataloader, eval_dataset, model, tokenizer, eval_when_training=False,pkl_file_path=None,language=None):
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        seq_inputs = batch[2].to(args.device)
        with torch.no_grad():
            cross_entropy_loss, logit = model(seq_ids=seq_inputs, input_ids=inputs, labels=label, language=language)
            lm_loss = cross_entropy_loss
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits[:, 0] > 0.5
    eval_acc = np.mean(labels == preds)
    eval_f1 = f1_score(labels, preds)
    eval_loss = eval_loss / nb_eval_steps
    eval_recall = recall_score(labels, preds)
    perplexity = torch.tensor(eval_loss)

    result = {
        "eval_recall": round(eval_recall, 4),
        "eval_loss": float(perplexity),
        "eval_acc": round(eval_acc, 4),
        "eval_f1": round(eval_f1, 4)
    }
    return result



def test(args, model, tokenizer, test_dataset, test_dataloader, language=None):
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(test_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in tqdm(test_dataloader, total=len(test_dataloader)):
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        seq_inputs = batch[2].to(args.device)
        with torch.no_grad():
            logit = model(seq_ids=seq_inputs, input_ids=inputs, language=language)
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    preds = logits[:, 0] > 0.5


    path_name = "_".join(args.fusion_languages)
    file_name = f"{path_name}--{language}_predictions.txt"
    output_path = os.path.join(args.output_dir, file_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(output_path, 'w') as f:
        for example, pred in zip(test_dataset.examples, preds):
            if pred:
                f.write(example.idx + '\t1\n')
            else:
                f.write(example.idx + '\t0\n')


def str2bool(str):
    return True if str.lower() == 'true' else False