#### finetune
##### training and predicting
export BERT_BASE_DIR=/home/hemei/joe/bert_feature/chinese_L-12_H-768_A-12 #全局变量
export MY_DATASET=/home/hemei/hbfeng/bert/data #全局变量

python run_classifier.py \
  --task_name=simqq \ #自己添加processor在processors字典里的key名
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \ #模型参数
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=2.0 \
  --output_dir=/tmp/simqq_output/ #模型输出路径


python run_classifier.py --task_name=simqq --do_train=true --do_eval=true --do_predict=true --data_dir=$MY_DATASET --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=128 --train_batch_size=32 --learning_rate=5e-5 --num_train_epochs=2.0 --output_dir=/tmp/simqq_output/


##### predicting with trained model
export TRAINED_CLASSIFIER=/tmp/simqq_output/

python run_classifier.py \
  --task_name=simqq \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=128 \
  --output_dir=/tmp/simqq_output/

python run_classifier.py --task_name=simqq --do_predict=true --data_dir=$MY_DATASET --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$TRAINED_CLASSIFIER --max_seq_length=128 --output_dir=/tmp/simqq_output/


#### feature-based

export BERT_BASE_DIR=/home/hemei/joe/bert_feature/chinese_L-12_H-768_A-12 #全局变量
export MY_DATASET=/home/hemei/hbfeng/bert/data/input.txt  #全局变量

python extract_features.py \
  --input_file=$MY_DATASET \
  --output_file=/tmp/output.jsonl \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --layers=-1,-2,-3,-4 \
  --max_seq_length=128 \
  --batch_size=8


