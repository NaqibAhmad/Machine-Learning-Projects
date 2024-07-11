import pandas as pd
from sklearn import linear_model
import warnings
warnings.filterwarnings('ignore')

datafile= pd.read_csv("C:/Users/Naqib Ahmad/Downloads/housepricesdataset.csv",sep=";")

print(datafile)

regmodel= linear_model.LinearRegression()
regmodel.fit(datafile[['area','roomcount','buildingage']],datafile['price'])

print('The price of the house is: ', regmodel.predict([[230,4,10]]))

