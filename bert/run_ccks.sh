#### finetune
##### training and predicting
export BERT_BASE_DIR=C:/chinese_L-12_H-768_A-12 #全局变量
export MY_DATASET=C:/dataset/wiki_qa/WikiQACorpus #全局变量


python run_classifier.py \
  --task_name=simqa \ #自己添加processor在processors字典里的key名
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
  --output_dir=/tmp/simqa_output/ #模型输出路径

python run_classifier.py --task_name=simqa --do_train=true --do_eval=true --do_predict=true --data_dir=$MY_DATASET --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=128 --batch_size=8 --learning_rate=5e-5 --num_train_epochs=2.0 --output_dir=/tmp/simqa_output/
