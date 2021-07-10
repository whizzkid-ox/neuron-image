from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from model import ann

# Author: Ryo Segawa (whizznihil.kid@gmail.com)

def dataload(row):
    # create neuron array
    neu = data[row, 1024:1525]
    # create image array
    image = data[row, 0:1024]
    return neu, image

# load image and neural data
data = np.loadtxt('PhDData_500n_1rep40000n_img.csv', delimiter=',')

#x_train = np.load("x_train.npy")
#y_train = np.load("y_train.npy")

# create train data arrays
total_train = sum([1 for _ in open('PhDData_500n_1rep40000n_img.csv')]) # 40000 rows
for i in range(total_train):
    x_temp, y_temp = dataload(i)
    if i == 0:
        x_train = x_temp
        y_train = y_temp
    else:
        x_train = np.vstack([x_train, x_temp])
        y_train = np.vstack([y_train, y_temp])
np.save("x_train", x_train)
np.save("y_train", y_train)


print("x_train's shape is", x_train.shape)
print("y_train's shape is", y_train.shape)

# number for CV
fold_num = 5 
seed = 7 # fix random seed for reproducibility
np.random.seed(seed)

# define X-fold cross validation
kf = KFold(n_splits=fold_num, shuffle=True, random_state=seed)
cvscores = []

## learn using training data
print("Start training!")
X = x_train
Y = y_train
batch_size = 128
epochs = 10
for train, test in kf.split(X, Y):
    # Fit the model
    print(X[train].shape)
    scores, model = ann(X[train],Y[train],X[test],Y[test],batch_size,epochs)
    cvscores.append(scores * 100)

# Check the cross validation
fig = plt.figure()
plt.scatter([x for x in range(fold_num)], cvscores)
plt.ylabel("SV score")
plt.title('Title')
fig.savefig("cv_score.png")
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

# Save model
print('save the model')
model.save('model.h5')


