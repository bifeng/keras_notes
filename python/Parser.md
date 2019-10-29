<https://blog.csdn.net/u013177568/article/details/62432761>



### action=’store_true’

在 demo1.py 中指定 action=’store_true’的时候： 

parser.add_argument(‘–is_train’, action=’store_true’, default=False)

在运行的时候： 
python demo1.py 默认是False 
python demo1.py –is_train 是True, 注意这里没有给 is_train赋值。



