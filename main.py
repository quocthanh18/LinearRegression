import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.1, 0.2, 0.3, 0.4, 0.5]).reshape(-1, 1) 
y = np.array([-0.18, 0.31, 1.03, 2.48, 3.73]).reshape(-1, 1)

def plot_data(x, y):
    plt.scatter(x, y, color='red', marker='.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

def preprocess(x, y):
    X = np.hstack((np.ones((x.shape[0], 1)), x, x ** 2))
    return X, y 

def compute_cost(x, y):
    X, y = preprocess(x, y)
    return np.linalg.inv(X.T @ X) @ X.T @ y

def plot_regression(x, y, w):
    plt.scatter(x, y, color='red', marker='.')
    plt.plot(x, w[0] + w[1] * x + w[2] * 1/2 * x ** 2, color='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.show()

plot_regression(x, y, compute_cost(x, y))
print(compute_cost(x, y))
# Một vật rơi theo phương thẳng đứng theo phương trình sau:
# s = s0 + v0t +
# 1
# 2
# gt2
# (1)
# Người ta thực hiện thí nghiệm thu được kết quả như sau:
# t 0.1 0.2 0.3 0.4 0.5
# Hãy xác định phương trình vật rơi theo phương thẳng đứng.