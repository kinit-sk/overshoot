import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define the new points
# points = [(0.7, 0), (0.73, 1), (0.76, 1), (0.79, 1), (0.81, 1), (0.84, 1), 
#           (0.86, 2), (0.88, 2), (0.9, 3), (0.92, 3), (0.94, 5), (0.95, 6), 
#           (0.96, 7), (0.97, 9), (0.98, 11), (0.985, 13), (0.99, 16)]


points = [(0.7, 0), (0.73, 1), (0.76, 1), (0.79, 1), (0.81, 1.0), (0.84, 1.5),
          (0.86, 1.5), (0.88, 2.0), (0.9, 2.5), (0.92, 3.5), (0.94, 4.7), (0.95, 5.7),
          (0.96, 6.9), (0.97, 8.700000000000001), (0.98, 11.500000000000002), (0.985, 13.300000000000002), (0.99, 15.700000000000003)]

# Separate the points into x and y coordinates
x, y = zip(*points)
x = np.array(x)
y = np.array(y)

# Define the exponential function to fit
def exponential_func(x, a, b):
    return a * np.exp(b * x)

# Fit the exponential function to the data
params, _ = curve_fit(exponential_func, x, y, p0=(1, 1))  # Initial guesses for a and b

# Generate y values for the exponential fit
x_fit = np.linspace(min(x), max(x), 100)
y_fit = exponential_func(x_fit, *params)

print(params)
# Plotting the original points and the fitted exponential curve
plt.plot(x, y, 'o', label='Data Points')
plt.plot(x_fit, y_fit, '-', label=f'Exponential Fit: y = {params[0]:.2f} * e^({params[1]:.2f} * x)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Least Mean Square Error Exponential Fit')
plt.legend()
plt.grid(True)
plt.show()