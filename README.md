# MNIST with TensorFlow-Slim

A Convolutional neural network for recognizing hand written digits. The script uses Matplotlib for visualizing layers, output, loss and accuracy.
---
### Model:

Input - 28 x 28

Convolutional layer - 28 x 28 filters: 24

Max pooling layer - 14 x 14

Convolutional layer - 14 x 14 filters: 200

Max pooling layer - 7 x 7

Convolutional layer - 7 x 7 filters: 20

Dropout layer - 0.8

Fully connected layer - 10 outputs (Softmax)

---

<img src="https://github.com/ErlendFax/MNIST-ConvNet-with-TensorFlow/blob/master/Img/figure_1.png" width="50%" height="50%">

<img src="https://github.com/ErlendFax/MNIST-ConvNet-with-TensorFlow/blob/master/Img/figure_4.png" width="50%" height="50%">

<img src="https://github.com/ErlendFax/MNIST-ConvNet-with-TensorFlow/blob/master/Img/figure_5.png" width="50%" height="50%">

<img src="https://github.com/ErlendFax/MNIST-ConvNet-with-TensorFlow/blob/master/Img/figure_6.png" width="50%" height="50%">

<img src="https://github.com/ErlendFax/MNIST-ConvNet-with-TensorFlow/blob/master/Img/figure_7.png" width="50%" height="50%">

<img src="https://github.com/ErlendFax/MNIST-ConvNet-with-TensorFlow/blob/master/Img/figure_8.png" width="50%" height="50%">

#### Test accuracy in this case: 0.985614:
<img src="https://github.com/ErlendFax/MNIST-ConvNet-with-TensorFlow/blob/master/Img/figure_9.png" width="50%" height="50%">

<img src="https://github.com/ErlendFax/MNIST-ConvNet-with-TensorFlow/blob/master/Img/figure_2.png" width="50%" height="50%">

<img src="https://github.com/ErlendFax/MNIST-ConvNet-with-TensorFlow/blob/master/Img/figure_3.png" width="50%" height="50%">
---
#### Few epochs and similar digits can result in bigger uncertainty:
<img src="https://github.com/ErlendFax/MNIST-ConvNet-with-TensorFlow/blob/master/Img/figure_10.png" width="50%" height="50%">
