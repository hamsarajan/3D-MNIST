# 3D-MNIST
A simple 2D Convolutional Neural Network for the classification of 3D MNIST dataset
### Required packages:
```
h5py
numpy
keras
sci-kit learn
matplotlib
```
### Training and Validation accuracy 
![Accuracy](/images/logo.png)
### Training and Validation loss 

### Accuracy of the trained model
The accuracy of the trained model is calculated using the code below:
```
predictions = model.predict(X_test)
predictions = np.argmax(predictions,axis = 1)
print(accuracy_score(y_test,predictions))
```
