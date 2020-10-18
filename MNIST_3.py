import h5py
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score

tf.reset_default_graph()
with h5py.File("full_dataset_vectors.h5", "r") as hf:
    X_train = hf["X_train"][:]
    y_train = hf["y_train"][:]
    X_test = hf["X_test"][:]
    y_test = hf["y_test"][:]

def array_to_color(array, cmap="Oranges"): # changing 4096 to 16,16,16
    s_m = plt.cm.ScalarMappable(cmap=cmap)
    return s_m.to_rgba(array)[:,:-1]
def rgb_data_transform(data): # creating a for loop to change 1d array to 3 channel data
    data_t = []
    for i in range(data.shape[0]):
        data_t.append(array_to_color(data[i]).reshape(16,16,16,3))
    return np.asarray(data_t, dtype=np.float32)

X_train = rgb_data_transform(X_train)
X_test = rgb_data_transform(X_test)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

epochs = 30
batch_size = 86
x_input  = tf.placeholder(tf.float32, shape=[None,16,16,16,3])
y_input = tf.placeholder(tf.int32, shape=[None,10])

conv1 = tf.layers.conv3d(inputs= x_input, filters=8, kernel_size=[3,3,3], activation=tf.nn.relu)
conv2 = tf.layers.conv3d(inputs= conv1, filters=16, kernel_size=[3,3,3], activation=tf.nn.relu)
pool3 = tf.layers.max_pooling3d(inputs=conv2, pool_size=[2,2,2], strides=2)
conv4 = tf.layers.conv3d(inputs= pool3, filters=32, kernel_size=[3,3,3], activation=tf.nn.relu)
conv5 = tf.layers.conv3d(inputs= conv4, filters=64, kernel_size=[3,3,3], activation=tf.nn.relu)
bn = tf.layers.batch_normalization(inputs=conv5, training=True)
pool6 = tf.layers.max_pooling3d(inputs=bn, pool_size=[2,2,2], strides=2)
dropout = tf.nn.dropout(pool6,keep_prob=0.25)
flatten = tf.reshape(bn, [-1,512])

W1 = tf.Variable(tf.random_normal([512, 128], stddev=0.01))
b1 = tf.Variable(tf.random_normal([128]))
W2 = tf.Variable(tf.random_normal([128,64], stddev=0.01))
b2 = tf.Variable(tf.random_normal([64]))
W3 = tf.Variable(tf.random_normal([64, 10], stddev=0.01))
b3 = tf.Variable(tf.random_normal([10]))

fc1 = tf.add(tf.matmul(flatten, W1), b1)
fc1 = tf.nn.relu(fc1)
dropout1 = tf.nn.dropout(fc1,keep_prob=0.5)
fc2 = tf.add(tf.matmul(dropout1,W2), b2)
fc2 = tf.nn.relu(fc2)
dropout2 = tf.nn.dropout(fc2,keep_prob=0.5)
y__ = tf.add(tf.matmul(dropout2, W3), b3)
y_ = tf.nn.softmax(y__)

loss = tf.losses.softmax_cross_entropy(y_input, y__)
optimiser = tf.train.AdamOptimizer().minimize(loss)
pred = tf.equal(tf.argmax(y_input,-1), tf.argmax(y_,-1))
accuracy = tf.reduce_mean(tf.cast(pred, tf.float32))

init = tf.global_variables_initializer()

# start the session

with tf.Session() as sess:
    sess.run(init)
    train_cost = []
    train_acc = []
    valid_acc = []
    valid_cost = []
    num_batch_train = int((len(X_train) / batch_size) + 1)
    num_batch_test = int((len(X_test) / batch_size) + 1)

    for epoch in range(epochs):
        cost = 0
        ac = 0
        ptr = 0
        for i in range(num_batch_train):
            print("{}/{}".format(i, num_batch_train))
            batch_x_train, batch_y_train = X_train[ptr:ptr + batch_size], y_train[ptr:ptr + batch_size]
            ptr += batch_size
            _,c,acc = sess.run([optimiser, loss, accuracy], feed_dict={x_input: batch_x_train, y_input: batch_y_train})
            cost += c / num_batch_train  # 1 loss = 32 images, 313 losses divided by total_batch = avg
            ac += acc/num_batch_train
        train_cost.append(cost)
        train_acc.append(ac)

        pointer = 0
        test_c = 0
        test_ac = 0
        for j in range(num_batch_test):
            batch_x_test, batch_y_test = X_test[pointer:pointer + batch_size], y_test[pointer:pointer + batch_size]
            pointer += batch_size
            _, c, acc = sess.run([optimiser, loss, accuracy], feed_dict={x_input: batch_x_test, y_input: batch_y_test})
            test_c += c / num_batch_test
            test_ac += acc / num_batch_test
        valid_cost.append(test_c)
        valid_acc.append(test_ac)
        print("Epoch:", (epoch + 1), "train_loss = {:.3f}, test_loss = {:.3f}, train_acc = {:.3f}, test_acc = {:.3f}  ".format(cost,test_c,ac,test_ac))

# Training and Validation loss curve
loss_train = train_cost
loss_val = valid_cost
plt.plot(loss_train, 'g', label='Training loss')
plt.plot(loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Training and validation accuracy curve
acc_train = train_acc
acc_val = valid_acc
plt.plot(acc_train, 'g', label='Training accuracy')
plt.plot(acc_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()