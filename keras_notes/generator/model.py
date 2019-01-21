import numpy as np
from keras.models import Sequential
from keras_notes.generator.generator import DataGenerator
path = './keras_notes/'

# Parameters
params = {'dim': (32,32,32),
          'batch_size': 64,
          'n_classes': 6,
          'n_channels': 1,
          'shuffle': True}

# Datasets
partition = {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']} # IDs
labels = {'id-1': 0, 'id-2': 1, 'id-3': 2, 'id-4': 1} # Labels

# id_1
id_1 = np.load(path + 'data/' + 'id-1' + '.npy')
# id_1.shape
# (32, 32, 32, 1)


# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

# Design model
model = Sequential()
[...] # Architecture
model.compile()

# Train model on dataset
model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    workers=6)