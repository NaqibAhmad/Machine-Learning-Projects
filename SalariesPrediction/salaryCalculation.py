import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df= pd.read_csv("C:/Users/Naqib Ahmad/Downloads/salaries_dataset.csv", sep=';')

print(df)

#plotting the graph of the salary
plt.scatter(df['experience_level'], df['salary'])
plt.xlabel('experience_level')
plt.ylabel('salary')
plt.savefig('graph.png', dpi=300)
plt.show('graph.png')

#using polynomial regression model
pol_reg= PolynomialFeatures(degree=5)
x_pol=pol_reg.fit_transform(df[['experience_level']])

lin_reg = LinearRegression()
lin_reg.fit(x_pol,df['salary'])


#generating result
y_head = lin_reg.predict(x_pol)
plt.plot(df['experience_level'], y_head, color='red', label='polynomial regression')
plt.legend()

plt.scatter(df['experience_level'], df['salary'])
plt.show()

x_pol= pol_reg.fit_transform([[4.5]])


print(lin_reg.predict(x_pol))