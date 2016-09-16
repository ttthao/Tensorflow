import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
#import input_data

# The MNIST digits data from 0-9
# The data is split into 55,000 training, 5,000 cross-validation and 10,000 data points test sets
# Each input (x's) is an 28x28 pixel image, flatten into a 784x1 tensor/vector (28 times 28)
# The training set (mnist.train.images) is a tensor (n-dim array) with a shape of [55000, 784]
# The first dim indexes the image and the second dim indexes the pixel intensity in of each pixel in an image
# The labels (y's) are the classes (digits 0-9) that represent the images
# The labels are represented as "1-hot vectors", which are vectors that are 0 in most dimensions
# and 1 in a single dimension i.e 3 would be [0,0,0,1,0,0,0,0,0,0]
# The label set's (mnist.train.labels) shape is [55000, 10],
# Where the first dim indexes 1-hot vectors for each corresponding image and
# the second dim indexes the value of dimensions for the 1-hot vector
print('Reading in input data...')
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# The input, None allows for any dimension value
# Or number of 784-pixel (28x28) examples
# Placeholders allow for user input when computation runs
print('Initializing input placeholder...')
x = tf.placeholder(tf.float32, [None, 784])

# The parameters
# Similar to theta from coursera
# 10 classes to predict from (0-9)
# Variables work like you'd think
print('Initializing parameters/weights variable...')
W = tf.Variable(tf.zeros([784, 10]))

# The bias for softmax regression
# Multi-class logistic regression
print('Initializing bias variable...')
b = tf.Variable(tf.zeros([10]))

# The hypothesis
# Bias is added to each calculation of the parameter/weight
print('Initializing hypothesis Wx + b')
y = tf.nn.softmax(tf.matmul(x, W) + b)

# The actual labels for the input
# Actual y/class (1-hot vectors)
print('Initializing actual labels/1-hot vector for each input (actual y/class)...')
y_ = tf.placeholder(tf.float32, [None, 10])

# The cost function used for softmax regression
# Different from the logistical regression cost function
print('Initializing cross-entropy cost function...')
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# Gradient descent to minimize cost
print('Running gradient descent with 0.01 step on cross-entropy...')
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# Initialize all the variables created
init = tf.initialize_all_variables()

# Create a new TF session (graph) and run the variable initialization
print('Initializing session...')
sess = tf.Session()
sess.run(init)

# Train with 1000 iterations
# Trains 100 training set examples at a time
# Batches data replaces placeholder values each iteration of gradient descent
# Batch GD = Stochastic GD
print('Begin algorithm training...')
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# Compare prediction to actual data
# argmax returns index of the highest entry in a tensor in a dimension
# Basically the highest probability value
# y is the algorithms prediction of the class/label, y_ is the actual data
# Returns a list of booleans
print('Comparing prediction results to actual data...')
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# Check prediction accuracy
# Cast to floating point numbers and take the average
print('Calculating accuracy (Should be around 91%)...')
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# Print accuracy on test set
print('Accuracy is: ')
print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
