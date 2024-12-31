import os
import argparse
import utils.parserTool.parse as ps
from utils.java_cfg import JAVA_CFG
from utils.python_cfg import PYTHON_CFG
from utils.php_cfg import PHP_CFG
from utils.c_cfg import C_CFG
from utils.parserTool.utils import remove_comments_and_docstrings
from utils.parserTool.parse import Lang
from utils.trainer import extract_pathtoken
import json
import pickle
import logging
import numpy as np
from transformers import (BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}

config_class, model_class, tokenizer_class = MODEL_CLASSES["roberta"]
tokenizer = tokenizer_class.from_pretrained("./models/codebert", do_lower_case=True)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, default=None, help="The language to be processed") 
    parser.add_argument("--data_file", default=None, type=str, required=True,help="The input data file.")
    parser.add_argument('--pkl_file', type=str, default='', help='for dataset path pkl file')
    args = parser.parse_args()
    output = open(os.path.join(args.pkl_file, f"{args.language}data.pkl"), 'wb')
    path_dict = {}
    num_id = 0     
    sum_ratio = 0  
    with open(os.path.join(args.data_file, f"{args.language}_train.jsonl")) as f:   
        for line in f:
            num_id += 1
            if num_id%100 == 0:
                print(num_id, flush=True)  
            js = json.loads(line.strip())  
            clean_code, code_dict = remove_comments_and_docstrings(js['func'], args.language)

            if args.language == "java":
                g = JAVA_CFG()
                code_ast = ps.tree_sitter_ast(clean_code, Lang.JAVA)
            elif args.language == "python":
                g = PYTHON_CFG()
                code_ast = ps.tree_sitter_ast(clean_code, Lang.PYTHON)
            elif args.language == "php":
                g = PHP_CFG()
                code_ast = ps.tree_sitter_ast(clean_code, Lang.PHP)
            elif args.language == "c":
                g = C_CFG()

            code_ast = ps.tree_sitter_ast(clean_code, Lang.C)
            s_ast = g.parse_ast_file(code_ast.root_node)
            num_path, cfg_allpath, _, ratio = g.get_allpath()
            path_tokens1 = extract_pathtoken(code_dict, cfg_allpath)
            sum_ratio += ratio
            path_dict[js['idx']] = path_tokens1, cfg_allpath
    print("train file finish...", flush=True)

    with open(os.path.join(args.data_file, f"{args.language}_valid.jsonl")) as f:      
        for line in f:
            num_id += 1
            if num_id%100==0:
                print(num_id, flush=True)
            js = json.loads(line.strip())
            clean_code, code_dict = remove_comments_and_docstrings(js['func'], args.language)

            if args.language == "java":
                g = JAVA_CFG()
                code_ast = ps.tree_sitter_ast(clean_code, Lang.JAVA)
            elif args.language == "python":
                g = PYTHON_CFG()
                code_ast = ps.tree_sitter_ast(clean_code, Lang.PYTHON)
            elif args.language == "php":
                g = PHP_CFG()
                code_ast = ps.tree_sitter_ast(clean_code, Lang.PHP)
            elif args.language == "c":
                g = C_CFG()
            code_ast = ps.tree_sitter_ast(clean_code, Lang.C)
            s_ast = g.parse_ast_file(code_ast.root_node)
            num_path, cfg_allpath, _, ratio = g.get_allpath()
            path_tokens1 = extract_pathtoken(code_dict, cfg_allpath)
            sum_ratio += ratio
            path_dict[js['idx']] = path_tokens1, cfg_allpath   
    print("valid file finish...", flush=True)

    with open(os.path.join(args.data_file, f"{args.language}_test.jsonl")) as f:   
        for line in f:
            num_id += 1
            if num_id%100==0:
                print(num_id, flush=True)
            js = json.loads(line.strip())
            clean_code, code_dict = remove_comments_and_docstrings(js['func'], args.language)

            if args.language == "java":
                g = JAVA_CFG()
                code_ast = ps.tree_sitter_ast(clean_code, Lang.JAVA)
            elif args.language == "python":
                g = PYTHON_CFG()
                code_ast = ps.tree_sitter_ast(clean_code, Lang.PYTHON)
            elif args.language == "php":
                g = PHP_CFG()
                code_ast = ps.tree_sitter_ast(clean_code, Lang.PHP)
            elif args.language == "c":
                g = C_CFG()
            code_ast = ps.tree_sitter_ast(clean_code, Lang.C)

            s_ast = g.parse_ast_file(code_ast.root_node)
            num_path, cfg_allpath, _, ratio = g.get_allpath()
            sum_ratio += ratio
            path_tokens1 = extract_pathtoken(code_dict, cfg_allpath)
            path_dict[js['idx']] = path_tokens1, cfg_allpath
    print("test file finish...", flush=True)
    print(sum_ratio/num_id, flush=True)
    pickle.dump(path_dict, output)
    output.close()

if __name__=="__main__":
    main()