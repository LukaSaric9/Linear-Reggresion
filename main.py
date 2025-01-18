import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('data.csv')

x = data.iloc[:, 0].values
y= data.iloc[:, 1].values

def fit(x_train, y_train): # Least squares method
    num = 0 # covariance between x_train and y_train
    den = 0 # variance of x_train
    k = 0 
    n = 0 
    for i in range(x_train.shape[0]):
        num = num + (y_train[i] - y_train.mean()) * (x_train[i] - x_train.mean())
        den = den + (x_train[i] - x_train.mean()) * (x_train[i] - x_train.mean())

    k = num / den
    n = y_train.mean() - (k * x_train.mean())
    return  k,n

def predict(x_test, k , n):
    return k * x_test + n

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 7)

k,n = fit(x_train, y_train)
print(f"Slope (k): {k}, Intercept (n): {n}")

predictions = [predict(x, k, n) for x in x_test]
for i in range(len(x_test)):
    print(f'Actual: {y_test[i]}, Predicted: {predictions[i]}')  

plt.scatter(x,y,color='blue', label='Data Points')
plt.plot(x,[k * xi + n for xi in x], color='red', label='Regression Line')
plt.legend()
plt.show()




