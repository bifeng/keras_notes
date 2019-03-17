from tensorflow_examples.stanford_tensorflow_tutorials.examples import utils
import tensorflow as tf
import matplotlib.pyplot as plt

path = 'C:\\Users\\kiss\\deep_coding_notes\\tensorflow_examples\\stanford_tensorflow_tutorials\\examples\\'
DATA_FILE = path + 'data\\birth_life_2010.txt'
data, n_samples = utils.read_birth_life_data(DATA_FILE)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

'''
notes: need to specify the shape
'''
# W = tf.get_variable(shape=1,initializer=tf.constant_initializer(), name='W')
# b = tf.get_variable(shape=1,initializer=tf.constant_initializer(), name='b')
W = tf.get_variable(initializer=tf.constant(0.0), name='W')
b = tf.get_variable(initializer=tf.constant(0.0), name='b')

pred = tf.multiply(X,W) + b
loss = tf.square(pred-Y)
'''
Huber loss - Robust to outliers
If the difference between the predicted value and the real value is small, square it
If it’s large, take its absolute value
'''
def huber_loss(pred, label, delta):
    residual = tf.abs(pred-label)
    fn1 = 0.5 * tf.square(residual)
    fn2 = tf.multiply(delta,residual) - 0.5 * tf.square(delta)
    loss = tf.cond(residual <= delta,true_fn=fn1,false_fn=fn2)
    return loss


learning_rate = 0.01
opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimize = opt.minimize(loss)

init = tf.global_variables_initializer()


'''
notes: attention the name conflict loss operation and loss_ result
'''
epoch = 50
with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graph/linear_reg', sess.graph)
    sess.run(init)
    for i in range(50):
        total_loss = 0
        for x,y in data:
            _,loss_ = sess.run([optimize,loss], feed_dict={X:x,Y:y})
            total_loss += loss_
    w_out,b_out = sess.run([W,b])
writer.close()


plt.plot(data[:,0], data[:,1], 'bo', label='Real data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, 'r', label='Predicted data')
plt.legend()
plt.show()