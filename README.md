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
![Accuracy](https://github.com/hamsarajan/3D-MNIST/blob/master/accuracy-128-256-128-epoch%3D100.png) 
### Training and Validation loss
![Loss](https://github.com/hamsarajan/3D-MNIST/blob/master/loss-128-256-128-epoch%3D100.png)

### Accuracy of the trained model
The accuracy of the trained model is calculated using the code below:
```
_, train_acc = model.evaluate(X_train, y_train, verbose=0)
_, test_acc = model.evaluate(X_test,y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
```
