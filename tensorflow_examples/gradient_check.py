import tensorflow as tf

'''
compute_gradients/apply_gradients
modify the gradients calculated by your optimizer
'''
# create an optimizer.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

# compute the gradients for a list of variables.
grads_and_vars = optimizer.compute_gradients(loss, <list of variables>)

# grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
# need to the 'gradient' part, for example, subtract each of them by 1.
subtracted_grads_and_vars = [(gv[0] - 1.0, gv[1]) for gv in grads_and_vars]

# ask the optimizer to apply the subtracted gradients.
optimizer.apply_gradients(subtracted_grads_and_vars)


'''
tf.stop_gradient
prevent certain tensors from contributing to the calculation of  the derivatives with respect to a specific loss
'''
tf.stop_gradient( input, name=None )


'''
tf.gradients
explicitly ask TensorFlow to calculate certain gradients
'''
x = tf.Variable(2.0)
y = 2.0 * (x ** 3)
z = 3.0 + y ** 2

grad_z = tf.gradients(z, [x, y])
with tf.Session() as sess:
	sess.run(x.initializer)
	print(sess.run(grad_z)) # >> [768.0, 32.0]
# 768 is the gradient of z with respect to x, 32 with respect to y


