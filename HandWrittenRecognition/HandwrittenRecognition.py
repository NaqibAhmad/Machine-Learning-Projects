import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

mnist = fetch_openml('mnist_784')

print(mnist.data.shape)

#defining methods to show imgaes from the mnist dataset

def showiimage(dframe, index):
    some_digit = dframe.to_numpy()[index]
    some_digit_image = some_digit.reshape(28,28)
    
    plt.imshow(some_digit_image, cmap="binary")
    plt.axis("off")
    plt.show()

#print(showiimage(mnist.data, 10))

#splitting data into training set and test set

train_img, test_img, train_lbl, test_lbl= train_test_split(mnist.data, mnist.target, test_size=1/7.0)

test_img_copy = test_img.copy()

#print(showiimage(test_img_copy, 2))

#Scaling unscaled data 
scaler = StandardScaler()

scaler.fit(train_img)

train_img = scaler.transform(train_img)
test_img= scaler.transform(test_img)

#applying the PCA algorithm

pca_model = PCA(.95)

pca_model.fit(train_img)
print(pca_model.n_components_)

test_img= pca_model.transform(test_img)
train_img = pca_model.transform(train_img)

#applying logistic regression
Log_reg= LogisticRegression(solver= "lbfgs", max_iter=10000)

#training data
Log_reg.fit(train_img, train_lbl)

#predicting letters
print(Log_reg.predict(test_img[40].reshape(1,-1)))

print(showiimage(test_img_copy, 40))

#measuring the accuracy of the model
print(Log_reg.score(test_img, test_lbl))