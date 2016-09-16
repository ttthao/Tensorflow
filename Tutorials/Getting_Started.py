import tensorflow as tf
import numpy as np

# Create 100 random x, y data points
x_data = np.random.rand(100).astype("float32")

# y = x * 0.1 + 0.3
y_data = x_data * 0.1 + 0.3

# Try to find values for W and b that compute y_data = W * x_data + b
# W Should be 0.1 and b should be 0.3

# Randomly initialize W (parameter/weight)
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# Set bias to 0
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

# Minimize the cost function
# Cost function is the mean square of the error between prediction and actual
loss = tf.reduce_mean(tf.square(y - y_data))

# Step is 0.5
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# Before starting, initialize the variables
init = tf.initialize_all_variables()

# Launch the graph
sess = tf.Session()
sess.run(init)

# Train the algorithm
for step in xrange(201):
    # Gradient descent on GD optimizer function
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(W), sess.run(b)

# Learns the best fit is roughly W: [0.1], b: [0.3]
