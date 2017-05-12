
# MNIST digit classification with Tensorflow

### Network Design :pencil:

#### Network 1:

 -  Layer 1 : 5x5 Convolution with 32 feature maps stride 1 
 - Layer 2 : Rectified Linear Unit :ReLU 
 - Layer 3 : Maxpooling 2x2 with stride 2 
 - Layer 4 : 5x5 Convolution with 64 feature maps stride 1 
 - Layer 5 : Rectified Linear Unit :ReLU  
 - Layer 6 : Maxpooling 2x2 with stride 2  
 - Layer 7 : Fully Connected Layer, and dropout with 50% prob.  
 - Layer 8: Fully Connected layer , output layer   

#### Network 2:
 
 - Layer 1 : 3x3 Convolution with 64 feature maps stride 1 
 - Layer 2 : Rectified Linear Unit :ReLU  
 - Layer 3 : Maxpooling 2x2 with stride 2  
 - Layer 4 : 3x3 Convolution with 128 feature maps stride 1  
 - Layer 5 : Rectified Linear Unit :ReLU  
 - Layer 6 : Maxpooling 2x2 with stride 2  
 - Layer 7 : Fully Connected Layer, and dropout with 50% prob.  
 - Layer 8: Fully Connected layer , output layer

### Experimental Results :bar_chart:
Following table shows the accuracy of networks for different data sets

DATA SET | Network 1 | Network 2
------------ | ------------- | -------------
28x28 Pure Data Set | 87,3 % | 89,1 %
14x14 Pure Data Set | 84,4 % | 86,0 %
14x14 Augmented Data Set | 88,2 % | 90,8 %
