#### finetune
##### training
# export BERT_BASE_DIR=C:/chinese_L-12_H-768_A-12 #全局变量
# export MY_DATASET=C:/dataset/wiki_qa/WikiQACorpus #全局变量
export BERT_BASE_DIR=/home/ap/nlp/hbfeng/chinese_L-12_H-768_A-12 #全局变量
export MY_DATASET=/home/ap/nlp/hbfeng/ccks_human/data/balance #全局变量
export MODEL_OUTPUT=/home/ap/nlp/hbfeng/ccks_human/model
export VALID_OUTPUT=/home/ap/nlp/hbfeng/ccks_human/result/valid
export DEVELOP_OUTPUT=/home/ap/nlp/hbfeng/ccks_human/result/develop

cpu:
python run_classifier_with_re.py --task_name=ccks_sim --do_train=true --do_eval=true --do_predict=true --data_dir=$MY_DATASET --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=128  --train_batch_size=8 --learning_rate=5e-5 --num_train_epochs=10.0 --output_dir=$MODEL_OUTPUT &
gpu:
python run_classifier_with_re.py --task_name=ccks_sim --do_train=true --do_eval=true --do_predict=true --data_dir=$MY_DATASET --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=128  --learning_rate=5e-5 --num_train_epochs=10.0 --output_dir=$MODEL_OUTPUT &

##### predicting with trained model
python run_classifier_with_ccks.py --task_name=ccks_sim --do_predict=true --data_dir=$MY_DATASET --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$MODEL_OUTPUT --max_seq_length=128 --output_dir=$VALID_OUTPUT &
python run_classifier_with_ccks_develop.py --task_name=ccks_sim --do_predict=true --data_dir=$MY_DATASET --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$MODEL_OUTPUT --max_seq_length=128 --output_dir=$DEVELOP_OUTPUT &

