import h5py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score

with h5py.File("full_dataset_vectors.h5", "r") as hf:
    X_train = hf["X_train"][:]
    y_train = hf["y_train"][:]
    X_test = hf["X_test"][:]
    y_test = hf["y_test"][:]

epochs = 100
batch_size = 32
x_input  = tf.placeholder(tf.float32, shape=[None, 4096])
y_input = tf.placeholder(tf.int32, shape=[None,])


W1 = tf.Variable(tf.random_normal([4096, 128], stddev=0.01))
b1 = tf.Variable(tf.random_normal([128]))
W2 = tf.Variable(tf.random_normal([128,256], stddev=0.01))
b2 = tf.Variable(tf.random_normal([256]))
W3 = tf.Variable(tf.random_normal([256,128], stddev=0.01))
b3 = tf.Variable(tf.random_normal([128]))
W4 = tf.Variable(tf.random_normal([128, 10], stddev=0.01))
b4 = tf.Variable(tf.random_normal([10]))

h1 = tf.add(tf.matmul(x_input, W1), b1)
h1 = tf.nn.sigmoid(h1)
dropout = tf.nn.dropout(h1,keep_prob=0.5)
h2 = tf.add(tf.matmul(dropout,W2), b2)
h2 = tf.nn.relu(h2)
dropout_1 = tf.nn.dropout(h2, keep_prob=0.5)
h3 = tf.add(tf.matmul(dropout_1,W3), b3)
h3 = tf.nn.relu(h3)
y__ = tf.add(tf.matmul(h3, W4), b4)
y_ = tf.nn.softmax(y__)

cross_entropy = tf.losses.sparse_softmax_cross_entropy(y_input, y__)
optimiser = tf.train.AdamOptimizer().minimize(cross_entropy)
init_op = tf.global_variables_initializer()

# start the session
train_cost = []
train_acc = []
valid_acc = []
valid_cost = []
with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    total_batch = int((len(X_train) / batch_size) + 1)
    for epoch in range(epochs):
        avg_cost = 0
        ptr = 0
        for i in range(total_batch):
            batch_x, batch_y = X_train[ptr:ptr + batch_size], y_train[ptr:ptr + batch_size]
            ptr += batch_size
            _, c = sess.run([optimiser, cross_entropy], feed_dict={x_input: batch_x, y_input: batch_y})
            avg_cost += c / total_batch
        train_cost.append(avg_cost)
        print("Epoch:", (epoch + 1), "loss =", "{:.3f}".format(avg_cost))
        c_valid = sess.run(cross_entropy, feed_dict={x_input: X_test, y_input: y_test})
        valid_cost.append(c_valid)

        pred_valid = sess.run(y_, feed_dict={x_input: X_test})
        pred_valid = np.argmax(pred_valid, axis=-1)
        accuracy_valid = accuracy_score(y_test, pred_valid)
        valid_acc.append(accuracy_valid)
        pred_train = sess.run(y_, feed_dict={x_input: X_train})
        pred_train = np.argmax(pred_train, axis=-1)
        accuracy_train = accuracy_score(y_train, pred_train)
        train_acc.append(accuracy_train)

#Training and Validation loss curve
loss_train = train_cost
loss_val = valid_cost
plt.plot(loss_train, 'g', label='Training loss')
plt.plot(loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


#Training and validation accuracy curve
acc_train = train_acc
acc_val = valid_acc
plt.plot(acc_train, 'g', label='Training accuracy')
plt.plot(acc_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()