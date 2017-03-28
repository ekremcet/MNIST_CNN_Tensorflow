# MNIST_CNN_Tensorflow
MNIST digit classification Convolutional Neural Network with Tensorflow

Network Design
Network 1:
	Layer 1 : 5x5 Convolution with 32 feature maps stride 1
	Layer 2 : Rectified Linear Unit :ReLU
	Layer 3 : Maxpooling 2x2 with stride 2
	Layer 4 : 5x5 Convolution with 64 feature maps stride 1
	Layer 5 : Rectified Linear Unit :ReLU
	Layer 6 : Maxpooling 2x2 with stride 2
	Layer 7 : Fully Connected Layer, and dropout with 50% prob.
	Layer 8: Fully Connected layer , output layer
Network 2:
	Layer 1 : 3x3 Convolution with 64 feature maps stride 1
	Layer 2 : Rectified Linear Unit :ReLU
	Layer 3 : Maxpooling 2x2 with stride 2
	Layer 4 : 3x3 Convolution with 128 feature maps stride 1
	Layer 5 : Rectified Linear Unit :ReLU
	Layer 6 : Maxpooling 2x2 with stride 2
	Layer 7 : Fully Connected Layer, and dropout with 50% prob.
	Layer 8: Fully Connected layer , output layer
    
