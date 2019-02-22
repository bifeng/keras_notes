from run_classifier import QQProcessor

dir = 'D:\\bert\\data\\'

pp = QQProcessor()
train = pp.get_train_examples(dir)
test = pp.get_test_examples(dir)

