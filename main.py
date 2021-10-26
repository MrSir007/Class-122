import cv2
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score as acc

# To fetch data from "openml" library
X, y = fetch_openml("mnist_784",version=1,return_X_y=True)
'''print(pd.Series(y).value_counts())'''
classes = ["0","1","2","3","4","5","6","7","8","9"]
nClass = len(classes)

# To display 5 sample from each object
samplePerObject = 5
# To set the size of the figure
figure = plt.figure(figsize=(nClass*2,(1+samplePerObject*2)))
# To create a "for loop" for the object
index_class = 0
for c in classes :
  indexes = np.flatnonzero(y==c)
  indexes = np.random.choice(indexes, samplePerObject, replace=False)

# To iterate over random index
a = 0
for i in indexes :
  pltIndex = a * nClass + index_class + 1
# To plot the image with heat map sample
heat = sb.heatmap(np.reshape(X[indexes], (28,28)), cmap=plt.cm.gray, xticklabels=False, yticklabels=False, cbar=False)
print(len(X))
print(X[0])