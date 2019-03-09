'''
1.
feed or fetch any graph nodes and variables (gradients, weight matrices and biases) using the unique names that
TensorFlow generates directly  sess.run("<name>:0", feed_dict={…}) even in the feed dictionary.
You can get the <name> part using the function tf.get_default_graph().as_graph_def() or
[str(op.name for op in tf.get_default_graph().get_operations()].

2. name scope with tensorboard

3. ...
 store references to the output tensors of the layers e.g. by appending them to a list layerOutputs.append(relu).
 Then you can access them e.g. by output1, output2 = sess.run([layerOutputs[1], layerOutputs[2]], feed_dict={…})

https://www.quora.com/How-does-one-access-the-intermediate-activation-layers-of-a-deep-net-in-a-TensorFlow
'''