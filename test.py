import numpy as np
from sklearn.linear_model import LinearRegression

# Dữ liệu thời gian t (giây)
time_data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

# Dữ liệu khoảng cách s (mét)
distance_data = np.array([-0.18, 0.31, 1.03, 2.48, 3.73])

# Biến đổi dữ liệu cho linear regression
x = time_data
y = distance_data

# Biến đổi thành ma trận dữ liệu đầu vào cho linear regression
X = np.column_stack((np.ones_like(x), x, 0.5 * x**2))

# Thực hiện linear regression
regressor = LinearRegression()
regressor.fit(X, y)

# Lấy giá trị tối ưu của các tham số s0, v0, và g
s0_optimal = regressor.intercept_
v0_optimal, g_optimal = regressor.coef_[1:]

print("Các tham số phù hợp nhất:")
print(f"s0_optimal = {s0_optimal}")
print(f"v0_optimal = {v0_optimal}")
print(f"g_optimal = {g_optimal}")
