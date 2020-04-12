import numpy as np 
import pandas as pd 
df=pd.read_csv("poojadata.csv")


from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
X=df['lead_time'].values.reshape(-1,1)
Y=df['adults']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
regressor = LinearRegression()  
regressor.fit(X_train, y_train)
p1 = regressor.predict([[101]])
p1[0]
