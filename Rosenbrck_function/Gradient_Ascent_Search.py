import numpy as np

def rosenbrock(x, y):
    return 100 * (y - x ** 2) ** 2 + (1 - x) ** 2

# 미분값
def gradient(x, y):
    grad_x = -400 * x * (y - x ** 2) - 2 * (1 - x)
    grad_y = 200 * (y - x ** 2)
    return np.array([grad_x, grad_y])

def gradient_descent(start_x, start_y, alpha, iterations):
    x, y = start_x, start_y
    x_history, y_history = [x], [y]

    for i in range(iterations):
        grad = gradient(x, y)
        x, y = x - alpha * grad[0], y - alpha * grad[1]
        x_history.append(x)
        y_history.append(y)

    return x_history, y_history