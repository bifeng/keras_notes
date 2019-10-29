'''
https://stackoverflow.com/questions/34775522/tensorflow-multiple-sessions-with-multiple-gpus

'''

'''
multiple session - you need separate running the session in each script
'''
import tensorflow as tf

W = tf.Variable(10)

sess1 = tf.Session()
sess2 = tf.Session()

sess1.run(W.initializer)
sess2.run(W.initializer)

print(sess1.run(W.assign_add(10)))  # >> 20
print(sess2.run(W.assign_sub(2)))  # >> 8

print(sess1.run(W.assign_add(100)))  # >> 120
print(sess2.run(W.assign_sub(50)))  # >> -42

sess1.close()
sess2.close()


'''
multiple graph
'''
g1 = tf.get_default_graph()
g2 = tf.Graph()
# add ops to the default graph
with g1.as_default():
    a = tf.constant(3)
# add ops to the user created graph
with g2.as_default():
    b = tf.constant(5)


'''
multiple graph
https://bretahajek.com/2017/04/importing-multiple-tensorflow-models-graphs/
'''
import tensorflow as tf

class ImportGraph():
    """  Importing and running isolated TF graph """
    def __init__(self, loc):
        # Create local graph and use it in the session
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            # Import saved model from location 'loc' into local graph
            saver = tf.train.import_meta_graph(loc + '.meta',
                                               clear_devices=True)
            saver.restore(self.sess, loc)
            # There are TWO options how to get activation operation:
              # FROM SAVED COLLECTION:
            self.activation = tf.get_collection('activation')[0]
              # BY NAME:
            self.activation = self.graph.get_operation_by_name('activation_opt').outputs[0]

    def run(self, data):
        """ Running the activation operation previously imported """
        # The 'x' corresponds to name of input placeholder
        return self.sess.run(self.activation, feed_dict={"x:0": data})


model_1 = ImportGraph('models/model_name')
model_2 = ImportGraph('models/different_model')

# Application of two different models, ploting results
import matplotlib.pylab as plt
data = range(101)
plt.plot(model_1.run(data), 'r')
plt.plot(model_2.run(data), 'g')
plt.plot(data, expected, 'o', markersize=2)
plt.show()