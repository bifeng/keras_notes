### data structure

#### model

##### Sequential model

a linear stack of layers

https://keras.io/models/sequential/



##### Model

arbitrary graphs of layers

[Keras functional API](https://keras.io/getting-started/functional-api-guide)



### build blocks

#### layer

keras.layers

#### optimizer

keras.optimizers

#### losses

keras.losses

#### metrics

```
# For custom metrics
import keras.backend as K

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)
```



#### utils

keras.utils.Sequence

multiprocessing



keras.utils.to_categorical





### debug

https://keras.io/callbacks/

#### LambdaCallback

#### Create a callback






















