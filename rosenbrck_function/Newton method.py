import numpy as np

# Rosenbrock 함수, 그래디언트, 헤시안 정의
def rosenbrock(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

# 그래디언트
def gradient(x, y):
    return np.array([-2*(1 - x) - 400*x*(y - x**2), 200*(y - x**2)])

# 헤시안
def hessian(x, y):
    return np.array([[2 - 400*y + 1200*x**2, -400*x], [-400*x, 200]])

# 뉴턴 방법 구현
def newtons_method(x0, y0, epsilon=1e-6, max_iter=100):
    xy = np.array([x0, y0])
    for _ in range(max_iter):
        grad = gradient(xy[0], xy[1])
        hess = hessian(xy[0], xy[1])
        xy_new = xy - np.linalg.inv(hess).dot(grad)
        if np.linalg.norm(xy_new - xy) < epsilon:
            break
        xy = xy_new
    return xy