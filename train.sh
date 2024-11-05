#!/bin/bash

python pretrain.py  --epoch 35 \
                    --output_dir=./results \
                    --model_type=roberta \
                    --data_file=./datasets \
                    --pkl_file=./utils/pkl_folder \
                    --tokenizer_name=./models/codebert \
                    --model_name_or_path=./models/codebert \
                    --block_size 400 --train_batch_size 5 --eval_batch_size 8 \
                    --learning_rate 2e-5 --max_grad_norm 1.0 --evaluate_during_training \
                    --seed 123456 --cnn_size 128 --filter_size 3 --d_size 128 \
                    \
                    --dataset=c_java_python_php \
                    \
                    --use_adapters=false \
                    --fusion_languages=c_php \
                    --target_language=php \

python finetune.py  --epoch 35 \
                    --output_dir=./results \
                    --model_type=roberta \
                    --data_file=./datasets \
                    --pkl_file=./utils/pkl_folder \
                    --tokenizer_name=./models/codebert \
                    --model_name_or_path=./models/codebert \
                    --block_size 400 --train_batch_size 5 --eval_batch_size 8 \
                    --learning_rate 2e-5 --max_grad_norm 1.0 --evaluate_during_training \
                    --seed 123456 --cnn_size 128 --filter_size 3 --d_size 128 \
                    \
                    --dataset=c_java_python_php \
                    \
                    --use_adapters=true \
                    --fusion_languages=c_php \
                    --target_language=php \
