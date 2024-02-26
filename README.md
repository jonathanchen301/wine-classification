# wine-classification

# Purpose

The goal of this project is to learn to implement a multilayer perceptron classifier using
PyTorch to predict what type of wine a wine is using multiple features.

Although the subgoal is to predict accurately based on multiple features, the main goal of this project is to successfully implement
a multilayer perceptron classifier using the PyTorch library.

# Classification, Softmax, and Argmax

Unlike regression, which I used in my medical insurance project, classification requires the use of a softmax and an argmax. In my forward
function, I have one hidden layer and an output layer. The output layer has 3 dimensions, one for each of the possible classes that the
model can predict. By softmaxing, it converts the outputs so that they add up to 1, which means each output is the probability that
the sample is that particular class. Then, when making predictions or evaluating the model, argmax is used which gives the index of the maximum value of the outputs post-softmax. The number given by the index is the class predicted by the model.

# Approach

First, I used PyTorch's DataLoader to load in the dataset. The DataLoader converts the labels into a tensor in which all indices are 0 except the index that is equal to the label, which is one. I then separated the data using an 80-10-10 split. I then implemented a
multilayer perceptron (MLP) to train the model. When evaluating or predicting, I applied softmax and argmax. For the MLP, I tuned the hyperparameters of learning rate, epochs, number of hidden layers, and number of dimensions within each hidden layer.

# Language + Libraries Used

Python | Libraries: pandas, torch, sklearn

# Results

After evaluating on the test set, the model was able to achieve 100% accuracy, successfully predicting every single class. Normally, this
would not happen, but the features and the problem was simple enough such that this was possible. I originally thought my model would overfit
the training data because I was using 32 hidden_dims, but found that the test accuracy was 100% later. Although this accuracy is amazing, I lost the opportunity to apply some regularization techniques like L2 regularization, dropout, or early stopping.

# Credit

Dataset from: [UC Irvine ML Repo](https://archive.ics.uci.edu/dataset/109/wine)