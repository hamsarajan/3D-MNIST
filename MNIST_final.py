import h5py
from tensorflow import keras
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
#from plotly.offline import iplot, init_notebook_mode
#import plotly.graph_objs as go

with h5py.File("full_dataset_vectors.h5", "r") as hf:
    X_train = hf["X_train"][:]
    y_train = hf["y_train"][:]
    X_test = hf["X_test"][:]
    y_test = hf["y_test"][:]

#Designing and training the CNN model
input1 = keras.layers.Input(shape=(4096,))
layer1 = keras.layers.Dropout(0.1)(input1)
layer2 = keras.layers.Dense(128, activation = "relu")(layer1)
layer3 = keras.layers.Dropout(0.5)(layer2)
layer4 = keras.layers.Dense(256, activation = "relu")(layer3)
layer5 = keras.layers.Dropout(0.5)(layer4)
layer6 = keras.layers.Dense(128, activation = "relu")(layer5)
layer7 = keras.layers.Dropout(0.5)(layer6)
output = keras.layers.Dense(10, activation = "softmax")(layer7)
model = keras.models.Model(inputs = input1, outputs = output)
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
history = model.fit(X_train, y_train, batch_size=32, epochs = 100 , validation_data=(X_test,y_test))

#Accuracy of the trained model on the test dataset
predictions = model.predict(X_test)
predictions = np.argmax(predictions,axis = 1)
#print(predictions)
#print(y_test)
#print(accuracy_score(y_test,predictions))

_, train_acc = model.evaluate(X_train, y_train, verbose=0)
_, test_acc = model.evaluate(X_test,y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

#Training and Validation loss curve
loss_train = history.history['loss']
loss_val = history.history['val_loss']
plt.plot(loss_train, 'g', label='Training loss')
plt.plot(loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Training and validation accuracy curve
acc_train = history.history['acc']
acc_val = history.history['val_acc']
plt.plot(acc_train, 'g', label='Training accuracy')
plt.plot(acc_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#with h5py.File("train_point_clouds.h5", "r") as points_dataset:
    #digits = []
    #for i in range(10):
        #digit = (points_dataset[str(i)]["img"][:],points_dataset[str(i)]["points"][:], points_dataset[str(i)].attrs["label"])
        #digits.append(digit)

#x_c = [r[0] for r in digits[0][1]]
#y_c = [r[1] for r in digits[0][1]]
#z_c = [r[2] for r in digits[0][1]]
#trace1 = go.Scatter3d(x=x_c, y=y_c, z=z_c, mode='markers',marker=dict(size=12, color=z_c, colorscale='Viridis', opacity=0.7))
#data = [trace1]
#layout = go.Layout(height=500, width=600, title="Digit: " + str(digits[0][2]) + " in 3D space")
#fig = go.Figure(data=data, layout=layout)
#iplot(fig)