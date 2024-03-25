import random

def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def hill_climbing(start_x, start_y, step_size, num_iterations):
    import random
    x, y = start_x, start_y

    for i in range(num_iterations):
        next_x = x + random.uniform(-step_size, step_size)
        next_y = y + random.uniform(-step_size, step_size)
        if rosenbrock(next_x, next_y) < rosenbrock(x, y):
            x, y = next_x, next_y

    return x, y, rosenbrock(x, y)