import torch
import numpy as np
import math
import random
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

class Particle:
    def __init__(self, dimension, num_clusters):
        self.position = np.random.rand(num_clusters, dimension)  # 각 군집 중심점의 위치
        self.velocity = np.random.rand(num_clusters, dimension)  # 속도 초기화
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

# Fitness function : (목적 함수)군집과 중심점 거리의 합 최소화
def fitness_function(centers, data):
    distances = euclidean_distances(data, centers)
    total_distance = sum([np.min(distances[i]) for i in range(distances.shape[0])])
    return total_distance

# Particle Swarm Optimization main algorithms
def pso(data, num_clusters, max_iter, num_particles, inertia_weight, c1, c2):
    num_features = data.shape[1]
    particles = [Particle(num_features, num_clusters) for _ in range(num_particles)]
    global_best_position = None
    global_best_fitness = float('inf')

    for _ in range(max_iter):
        for particle in particles:    # pbest
            fitness = fitness_function(particle.position, data)
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position.copy()

        for particle in particles:    # gbest
            r1 = np.random.rand(num_clusters, num_features)
            r2 = np.random.rand(num_clusters, num_features)

            cognitive_component = c1 * r1 * (particle.best_position - particle.position)
            social_component = c2 * r2 * (global_best_position - particle.position)
            particle.velocity = inertia_weight * particle.velocity + cognitive_component + social_component
            particle.position += particle.velocity

    km = KMeans(n_clusters=num_clusters, init=global_best_position, n_init=1)
    km.fit(data)
    silhouette_avg = silhouette_score(data, km.labels_)
    return km.labels_, silhouette_avg

def initialize_clusters(num_clusters, data):
    return np.random.rand(num_clusters, data.shape[1])

def roulette_wheel_selection(population, fitness_scores):
    inverse_fitness = 1.0 / np.array(fitness_scores)
    total_inverse_fitness = np.sum(inverse_fitness)
    selection_probabilities = inverse_fitness / total_inverse_fitness
    selected_indices = np.random.choice(range(len(population)), size=len(population) // 2, replace=False, p=selection_probabilities)
    return [population[i] for i in selected_indices]

def crossover(parent1, parent2, crossover_rate):
    if np.random.rand() > crossover_rate:
        return parent1.copy(), parent2.copy()

    crossover_point = random.randint(0, parent1.shape[0] - 1)
    child1 = np.vstack((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.vstack((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

def mutation(centers, data, mutation_rate):
    if np.random.rand() > mutation_rate:
        return centers

    mutation_point = random.randint(0, centers.shape[0] - 1)
    centers[mutation_point] = data[random.randint(0, data.shape[0] - 1)]
    return centers

# Genetic Algorithms Main
def ga(data, num_clusters, num_generations, population_size, crossover_rate, mutation_rate):
    population = [initialize_clusters(num_clusters, data) for _ in range(population_size)]
    best_solution = None
    best_fitness = float('inf')

    for generation in range(num_generations):
        fitness_scores = [fitness_function(clusters, data) for clusters in population]

        if min(fitness_scores) < best_fitness:
            best_fitness = min(fitness_scores)
            best_solution = population[np.argmin(fitness_scores)]

        selected_population = roulette_wheel_selection(population, fitness_scores)

        next_population = []
        while len(next_population) < population_size:
            parent1, parent2 = random.sample(selected_population, 2)
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            next_population.append(mutation(child1, data, mutation_rate))
            if len(next_population) < population_size:
                next_population.append(mutation(child2, data, mutation_rate))

        population = next_population

    km = KMeans(n_clusters=num_clusters, init=best_solution, n_init=1)
    km.fit(data)
    silhouette_avg = silhouette_score(data, km.labels_)
    return km.labels_, silhouette_avg

class SimulatedAnnealing:
    def __init__(self, data, num_clusters, max_iter, initial_temp, cooling_rate, stop_threshold=1e-5):
        self.data = data
        self.num_clusters = num_clusters
        self.temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        self.num_features = data.shape[1]
        self.centroids = self.initialize_centroids()
        self.best_centroids = self.centroids.copy()
        self.best_fitness = self.fitness_function(self.centroids)
        self.stop_threshold = stop_threshold

    def initialize_centroids(self):
        """Initialize centroids within the range of the data."""
        return np.random.rand(self.num_clusters, self.data.shape[1])

    def fitness_function(self, centers):
        """Calculate the inertia (sum of squared distances) for the given centroids."""
        distances = euclidean_distances(self.data, centers)
        total_distance = sum([np.min(distances[i]) for i in range(distances.shape[0])])
        return total_distance

    def get_neighbor(self, centroids, step_size=0.1):
        """Generate neighbor centroids."""
        neighbor = centroids + np.random.uniform(-step_size, step_size, centroids.shape)
        min_vals = np.min(self.data, axis=0)
        max_vals = np.max(self.data, axis=0)
        for i in range(self.num_clusters):
            neighbor[i] = np.clip(neighbor[i], min_vals, max_vals)
        return neighbor

    def simulated_annealing(self):
        """Run the simulated annealing algorithm."""
        current_solution = self.centroids
        current_cost = self.fitness_function(current_solution)
        iterations = 1

        while self.temp > self.stop_threshold and iterations < self.max_iter:
            neighbor = self.get_neighbor(current_solution)
            neighbor_cost = self.fitness_function(neighbor)

            cost_difference = neighbor_cost - current_cost

            # Improving move + Non-improving move
            if cost_difference < 0 or math.exp(-cost_difference / self.temp) > random.random():
                current_solution = neighbor
                current_cost = neighbor_cost

                if current_cost < self.best_fitness:
                    self.best_fitness = current_cost
                    self.best_centroids = current_solution

            self.temp *= self.cooling_rate
            iterations += 1

        return self.best_centroids

    def run(self):
        """Run simulated annealing and return the labels and silhouette score."""
        best_centroids = self.simulated_annealing()
        kmeans = KMeans(n_clusters=self.num_clusters, init=best_centroids, n_init=1)
        kmeans.fit(self.data)
        silhouette_avg = silhouette_score(self.data, kmeans.labels_)
        return kmeans.labels_, silhouette_avg


def sk_means(data, num_clusters):
    km = KMeans(n_clusters=num_clusters, random_state=42)
    km.fit(data)
    silhouette_avg = silhouette_score(data, km.labels_)
    return km.labels_, silhouette_avg
