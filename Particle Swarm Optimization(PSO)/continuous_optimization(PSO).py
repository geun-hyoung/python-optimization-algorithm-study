import numpy as np

def object_function(x, y):
    return (4 - 2.1 * x ** 2 + (x ** 4) / 3) * x ** 2 + x * y + (-4 + 4 * y ** 2) * y ** 2

def pso_algorithm(x_bounds, y_bounds, num_particles=30, max_iter=100, w=0.5, c1=1.0, c2=1.0):
    particle_positions = np.random.rand(num_particles, 2)
    particle_positions[:, 0] = particle_positions[:, 0] * (x_bounds[1] - x_bounds[0]) + x_bounds[0]
    particle_positions[:, 1] = particle_positions[:, 1] * (y_bounds[1] - y_bounds[0]) + y_bounds[0]

    particle_velocities = np.random.rand(num_particles, 2) - 0.5

    pbest_positions = particle_positions.copy()
    pbest_scores = np.array([float('inf') for _ in range(num_particles)])

    gbest_score = float('inf')
    gbest_position = np.array([0, 0])

    for iteration in range(max_iter):
        for i in range(num_particles):
            # 적합도 평가
            fitness = object_function(particle_positions[i, 0], particle_positions[i, 1])

            if fitness < pbest_scores[i]:
                pbest_scores[i] = fitness
                pbest_positions[i] = particle_positions[i].copy()

            if fitness < gbest_score:
                gbest_score = fitness
                gbest_position = particle_positions[i].copy()

        for i in range(num_particles):
            r1, r2 = np.random.rand(2)
            velocity_update = (w * particle_velocities[i] +
                               c1 * r1 * (pbest_positions[i] - particle_positions[i]) +
                               c2 * r2 * (gbest_position - particle_positions[i]))
            particle_positions[i] += velocity_update
            particle_velocities[i] = velocity_update

        print(f"Iteration {iteration+1}/{max_iter} - Best Fitness: {gbest_score}")

    return gbest_position[0], gbest_position[1], gbest_score

if __name__ == "__main__":
    x_bounds = [-3.0, 3.0]
    y_bounds = [-2.0, 2.0]
    best_x, best_y, best_fitness = pso_algorithm(x_bounds, y_bounds)
    print(f"Best solution: x={best_x}, y={best_y}, Best fitness: {best_fitness}")