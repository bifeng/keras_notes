#### [Tensorflow: Cannot interpret feed_dict key as Tensor](https://stackoverflow.com/questions/40785224/tensorflow-cannot-interpret-feed-dict-key-as-tensor)

```py
from keras import backend as K

#Before prediction
K.clear_session()

#After prediction
K.clear_session()
```

