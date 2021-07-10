import pandas as pd
import numpy as np
from PIL import Image
import cv2

# load data
def dataload(row):
    dt = np.loadtxt('PhDData_500n_1rep40000n_img.csv', delimiter=',')
    dt = dt[row, 0:1024]
    np.save("data", dt)
    return dt


#data = dataload(1) # argument is any num of row corresponding to an image in the .csv
data = np.load("data.npy")
data = np.reshape(data,(32, 32))
print(data.ndim)

data = data*255 # to avoid float pixels
data = data.astype(np.uint8)
image = Image.fromarray(data)
#image = image.convert('RGB')
#image = image.convert("L")
image.save("image.jpg")

# resize image
img_resize = image.resize((256, 256))
img_resize.save('image_resize.jpg')
