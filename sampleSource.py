# From @minsuk-heo

# Imports
import tensorflow as tf
import numpy as np

# Collect MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# * Data Features
# x_train has 60,000 samples
# - Each sample has 28x28 pixels, with each pixel value going from 0 to 255

# Split training data into both training and validation data (for early stop)
x_val = x_train[50000:60000]
x_train = x_train[0:50000]  # This split fixes overfitting
y_val = y_train[50000:60000]
y_train = y_train[0:50000]

# Reshape into single vector (horizontal, then vertical - 28x28 px)
x_train = x_train.reshape(50000, 784)
x_val = x_val.reshape(10000, 784)
x_test = x_test.reshape(10000, 784)

# Normalize the data from 0 -> 100
x_train = x_train.astype("float32")
x_train /= 255
x_val = x_val.astype("float32")
x_val /= 255
x_test = x_test.astype("float32")
x_test /= 255

# Label to one-hot encoding value (array where 1 is proper output)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_val = tf.keras.utils.to_categorical(y_val, 10)
y_test = tf.keras.utils.to_categorical(y_val, 10)

# Implement Tensorflow MLP graph
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])


def mlp(x):
    # Hidden layer 1
    w1 = tf.Variable(tf.random_uniform([784, 256]))
    b1 = tf.Variable(tf.zeros([256]))
    h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
    # Hidden layer 2
    w2 = tf.Variable(tf.random_uniform([256, 128]))
    b2 = tf.Variable(tf.zeros([128]))
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
    # Output layer
    w3 = tf.Variable(tf.random_uniform([128, 10]))
    b3 = tf.Variable(tf.zeros([10]))
    logits = tf.matmul(h2, w3) + b3

    return logits


logits = mlp(x)

loss_op = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y))

train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss_op)

# Perform implementation step
# Init.
init = tf.global_variables_initializer()

# Train hyperparameters
epoch_cnt = 30
batch_size = 1000
iteration = len(x_train) // batch_size

# Start training
with tf.Session() as sess:
    # Run initializer
    sess.run(init)
    for epoch in range(epoch_cnt):
        avg_loss = 0.
        start = 0
        end = batch_size

        for i in range(iteration):
            _, loss = sess.run([train_op, loss_op],
                               feed_dict={x: x_train[start: end], y: y_train[start: end]})
            start += batch_size
            end += batch_size
            # Compute average loss
            avg_loss += loss / iteration

        # Validate model
        preds = tf.nn.softmax(logits)  # Apply softmax to logits
        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        cur_val_acc = accuracy.eval({x: x_val, y: y_val})
        print("epoch: "+str(epoch)+", validation accuracy: "
              + str(cur_val_acc) + ', loss: '+str(avg_loss))

    # Test model
    preds = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("[Test Accuracy] :", accuracy.eval({x: x_test, y: y_test}))
