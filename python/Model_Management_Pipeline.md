[Model Management Systems: Scikit-Learn and Django](https://us.pycon.org/2017/schedule/presentation/318/) [ppt](https://github.com/hanneshapke/pycon2017posters)

[WEB APPS THAT LEARN: AN ARCHITECTURE FOR MACHINE LEARNING IN DJANGO](https://us.pycon.org/2016/schedule/presentation/1614/) [ppt](https://github.com/DistrictDataLabs/PyCon2016/tree/master/posters/ddl-machine-learning)

https://2017.djangocon.us/talks/using-django-docker-and-scikit-learn-to-bootstrap-your-machine-learning-project/





功能：

1 - 可回退到历史版本

2 - 可预测

3 - 可训练

4 - 形成闭环





API

Predict



Train (retrain/increment_train)





Model

|                                            |                         |      |
| ------------------------------------------ | ----------------------- | ---- |
| model_id                                   | 唯一标识                |      |
| model_name                                 | 模型名称                |      |
| model_description                          | 模型描述 - 解决什么问题 |      |
| model_version                              | 模型版本                |      |
| model_file                                 | 模型文件                |      |
| model_config(random number/hyperparameter) | 模型配置文件            |      |
| model_metrics                              | 模型指标                |      |





Preprocess

preprocess_config(tokenizer/stop_words/new_words...) 





Data

| input_data | label_1 - 分类 (using prob select hard example) | label_2 - 排序 (using prob select hard example) |
| ---------- | ----------------------------------------------- | ----------------------------------------------- |
|            |                                                 |                                                 |
|            |                                                 |                                                 |
|            |                                                 |                                                 |

train_data/test_data - 标准数据集

| predict_data | label (using prob select hard example) |      |
| ------------ | -------------------------------------- | ---- |
|              |                                        |      |
|              |                                        |      |
|              |                                        |      |

predict_data - 各业务线各时间段的预测指标统计    反馈 







