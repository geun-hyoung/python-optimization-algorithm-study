import numpy as np
import random
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
random.seed(42)
class Particle:
    def __init__(self, dimension, num_clusters):
        self.position = np.random.rand(num_clusters, dimension)  # 각 군집 중심점의 위치
        self.velocity = np.random.rand(num_clusters, dimension)  # 속도 초기화
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')

# Fitness function 수정: 클러스터 내 거리의 합 최소화
def fitness_function(centers, data):
    distances = np.sqrt(((data[:, np.newaxis, :] - centers) ** 2).sum(axis=2))
    closest = np.argmin(distances, axis=1)
    inertia = sum([np.min(distances[i]) for i in range(distances.shape[0])])
    return inertia

def pso(data, num_clusters, max_iter, num_particles, inertia_weight, c1, c2):
    num_features = data.shape[1]
    particles = [Particle(num_features, num_clusters) for _ in range(num_particles)]
    global_best_position = None
    global_best_fitness = float('inf')

    for _ in range(max_iter):
        for particle in particles:
            fitness = fitness_function(particle.position, data)
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particle.position.copy()

        for particle in particles:
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
    num_features = data.shape[1]
    centers = np.zeros((num_clusters, num_features))
    for i in range(num_features):
        min_val, max_val = np.min(data[:, i]), np.max(data[:, i])
        centers[:, i] = np.random.uniform(min_val, max_val, size=num_clusters)
    return centers

def calculate_intra_cluster_distance(centers, data):
    distances = euclidean_distances(data, centers)
    total_distance = sum([np.min(distances[i]) for i in range(distances.shape[0])])
    return total_distance

def roulette_wheel_selection(population, fitness_scores):
    inverse_fitness = 1.0 / np.array(fitness_scores)
    total_inverse_fitness = np.sum(inverse_fitness)
    selection_probabilities = inverse_fitness / total_inverse_fitness
    selected_indices = np.random.choice(range(len(population)), size=len(population) // 2, replace=False,
                                        p=selection_probabilities)

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

# 유전 알고리즘 메인 함수
def ga(data, num_clusters, num_generations, population_size, crossover_rate, mutation_rate):
    population = [initialize_clusters(num_clusters, data) for _ in range(population_size)]
    best_solution = None
    best_fitness = float('inf')

    for generation in range(num_generations):
        fitness_scores = [calculate_intra_cluster_distance(clusters, data) for clusters in population]

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
    def __init__(self, data, num_clusters, max_iter, initial_temp, cooling_rate, stop_threshold=1):
        self.data = data
        self.num_clusters = num_clusters
        self.temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        self.num_features = data.shape[1]
        self.centroids = np.random.rand(num_clusters, self.num_features)
        self.best_centroids = self.centroids.copy()
        self.best_fitness = self.fitness_function(self.centroids)
        self.stop_threshold = stop_threshold

    def fitness_function(self, centers):
        distances = np.sqrt(((self.data[:, np.newaxis, :] - centers) ** 2).sum(axis=2))
        closest = np.argmin(distances, axis=1)
        inertia = sum([np.min(distances[i]) for i in range(distances.shape[0])])
        return inertia

    def perturb_centroids(self):
        new_centroids = self.centroids.copy()
        for i in range(self.num_clusters):
            if np.random.rand() < 0.5:
                new_centroids[i] += np.random.normal(0, 1, self.num_features)
            else:
                new_centroids[i] -= np.random.normal(0, 1, self.num_features)
        return new_centroids

    def accept_probability(self, old_fitness, new_fitness):
        if new_fitness < old_fitness:
            return 1.0
        return np.exp((old_fitness - new_fitness) / self.temp)

    def run(self):
        for _ in range(self.max_iter):
            new_centroids = self.perturb_centroids()
            new_fitness = self.fitness_function(new_centroids)
            if self.accept_probability(self.best_fitness, new_fitness) > np.random.rand():
                self.centroids = new_centroids
                if new_fitness < self.best_fitness:
                    if abs(self.best_fitness - new_fitness) < self.stop_threshold:
                        break
                    self.best_fitness = new_fitness
                    self.best_centroids = new_centroids
            self.temp *= self.cooling_rate

        km = KMeans(n_clusters=self.num_clusters, init=self.best_centroids, n_init=1)
        km.fit(self.data)
        silhouette_avg = silhouette_score(self.data, km.labels_)
        return km.labels_, silhouette_avg