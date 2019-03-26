import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sympy import *

# Question1:
print("Question1")

#data:
Deposit = np.array([2,5,10,20,25,30]).reshape(-1,1)
Returned = np.array([72,103,170,296,406,449])
Pro_returned = Returned/500
print(Pro_returned)

#Plot
f1=plt.figure(1)
plt.title("Estimated proportions:")
plt.xlabel("Deposit X")
plt.ylabel("Returned Probability")

plt.scatter(Deposit,Pro_returned)
#plt.show()


#logistic regression
return_class = Pro_returned.copy()
return_class[return_class>0.5]=1
return_class[return_class<=0.5]=0
print(return_class)

lr = LogisticRegression()
lr.fit(Deposit,return_class)

print(lr.coef_,lr.intercept_)
predict_prob = lr.predict_proba(Deposit)
print("predict_prob\n",predict_prob)

# estimate probability that a book will be returned when the deposit is 15 cents:
prob_15 = lr.predict_proba(np.array([15]).reshape(-1,1))
print("prob_15:\n",prob_15)


#solve prob=75%
x = Symbol("x")
sol = solve([log(0.75/0.25)+0.94570484-0.1205373*x],[x])
print("solve prob=75%\n",sol)


# Question 2:

#data:
year = np.array([0,1,2,3,4,5,6,7,8,9]).reshape(-1,1)
sales = np.array([98,135,162,178,221,232,283,300,374,395])

#plot
plt.figure(2)
plt.title("Annual sales:")
plt.xlabel("# year")
plt.ylabel("sales (k)")

plt.scatter(year,sales)
#plt.show()

#transform
sales_trans = sales**(1/2)
print(sales_trans)

#linear regression
linreg = LinearRegression().fit(year,sales_trans)
print("coefficient:", linreg.coef_)
print("intercept:", linreg.intercept_)
x = np.linspace(0,10)
y = linreg.intercept_ + linreg.coef_*x

# plot line and transformed data:
plt.figure(3)
plt.title("Linear regression:")
plt.xlabel("# year")
plt.ylabel("transformed sales")
plt.scatter(year,sales_trans)
plt.plot(x,y)
#plt.show()

#prediction:
predictions = linreg.intercept_ + linreg.coef_*year
predictions = predictions.reshape(1,-1)
print("predictions:\n", predictions)

#residuals:
resid = sales_trans -predictions
print("residuals\n", resid)