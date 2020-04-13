import numpy as np 
import pandas as pd 
df=pd.read_csv("poojadata.csv")

from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split 
X=df[['lead_time']]
Y=df[['adults']]
X_train,X_test,y_train,y_test= train_test_split(X, Y, test_size = 0.2, random_state=42) 
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
p1=knn.predict([['69']])
p1[0]
