Author: Ryo Segawa (whizznihil.kid@gmail.com)<br>
<br>
In this task, an image is estimated from neural data.<br>
<br>
<br>
Training dataset:<br>
Responses of 500 neurons to 40000 different images taken from a grayscale version of the CIFAR100 dataset (https://www.cs.toronto.edu/~kriz/cifar.html).<br>
The magnitude of the response can be thought of as the number of action potentials fired by each neuron when the image is presented.<br>
The file containing each image and the magnitude of the response of each of the 500 neurons is: PhDData_500n_1rep40000n_img.csv (must be in the same directory)<br>
<br>
The responses of each image and neuron set make up one line of the csv. <br>
The first 1024 elements of each line are the images of pixel values presented to the neurons. <br>
When this is formatted into a 32x32 matrix, the image presented to the neuron is displayed. <br>
The next 500 elements in a row are the responses of the 500 neurons. <br>
Each column contains the responses of the same neuron. <br>
Thus, the element (i, j+1024) represents the response of neuron j to image i.<br>
<br>
<br>
Test Data Set:<br>
Another simulated dataset of the responses of the same 500 neurons to 100 new images, but in this case no images are provided.<br>
PhD_neuron_output.csv<br>
<br>
The task can be divided into two steps: learning and prediction.<br>
There are two executables, task1.py and task2.py.<br>
<br>
<br>
・task1.py<br>
This script trains an artificial neural network (ANN) on a training dataset, where the input to the ANN is a neuron response and the output is an image. Cross-validation is performed here.<br>
-link: model.py<br>
The script that constructs the ANN model. If you want to make any changes to the model, you can modify it.<br>
Input: training dataset (PhDData_500n_1rep40000n_img.csv)<br>
Output: model (model.h5), results of cross-validation (cv_score.png)<br>
<br>
<br>
・task2.py<br>
This script uses the ANN model learned in task1.py to estimate the image for the neuron responses in the test dataset.<br>
Input: Test dataset (PhD_neuron_output.csv)<br>
Output: Image for an arbitrary neuron response (image_resize_*.jpg)<br>
