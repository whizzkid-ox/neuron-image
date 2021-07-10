import numpy as np
import keras
from keras.models import load_model
from PIL import Image

# Author: Ryo Segawa (whizznihil.kid@gmail.com)

# load neural data
data_test = np.loadtxt('PhD_neuron_output.csv', delimiter=',')

x_test = np.load("x_test.npy")


def dataload_test(row):
    # create neuron array
    neu = data_test[row, 0:500]
    return neu

# create data array
total_test = 10#sum([1 for _ in open('PhD_neuron_output.csv')]) # 100 rows 
for i in range(total_test):
    x_temp = dataload_test(i)
    if i == 0:
        x_test = x_temp
    else:
        x_test = np.vstack([x_test, x_temp])
np.save("x_test", x_test)

print("x_test's shape is", x_test.shape)


## apply the test data to the model
model = load_model('model.h5')
model.summary()
output = model.predict(x_test)

## Output as image
# Choose i you want to see as an image between [0,100]
i = 0
data = np.reshape(output[i],(32, 32))
data = data*255 # to avoid float pixels
data = data.astype(np.uint8)

image = Image.fromarray(data)
image.save("image_"+str(i)+".jpg")

# resize image
img_resize = image.resize((256, 256))
img_resize.save("image_resize_"+str(i)+".jpg")