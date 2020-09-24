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
![Accuracy](https://github.com/hamsarajan/3D-MNIST/blob/master/Training%20and%20Validation%20accuracy.png) 
### Training and Validation loss
![Loss](https://github.com/hamsarajan/3D-MNIST/blob/master/Training%20and%20Validation%20loss.png)

## Early Stopping
From the graphs, it is clear that the model is overfitting and therefore, I used early stopping method to reduce overfitting

### Training and Validation accuracy and loss after Early Stopping
![Accuracy_es](https://github.com/hamsarajan/3D-MNIST/blob/master/Early%20stopping_accuracy.png)
![loss_es](https://github.com/hamsarajan/3D-MNIST/blob/master/Early%20stopping_loss.png)
### Accuracy of the trained model
The accuracy of the trained model is calculated using the code below:
```
predictions = model.predict(X_test)
predictions = np.argmax(predictions,axis = 1)
print(accuracy_score(y_test,predictions))
```
