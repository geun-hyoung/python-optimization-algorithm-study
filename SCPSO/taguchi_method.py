import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools

import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder

from text_vectorization import spectral_embedding
from algorithms import pso, ga, SimulatedAnnealing

def run_experiment(algorithm, params, data, k, max_iter=100):
    if algorithm == 'pso':
        labels, score = pso(data, k, max_iter, **params)
    elif algorithm == 'ga':
        labels, score = ga(data, k, **params)
    elif algorithm == 'sa':
        sa = SimulatedAnnealing(data, k, max_iter, **params)
        labels, score = sa.run()
    else:
        raise ValueError("Unsupported algorithm")
    return score

def calculate_sn_ratio_minimization(scores):
    n = len(scores)
    sum_of_squares = np.sum(np.array(scores) ** 2)
    sn_ratio = -10 * np.log10(sum_of_squares / n)
    return sn_ratio

def create_random_combinations(params_grid, num_samples):
    all_combinations = list(itertools.product(*params_grid.values()))
    random_combinations = random.sample(all_combinations, num_samples)
    return random_combinations

def run_taguchi_experiment(algorithm, params_grid, data, k, max_iter, random_combinations):
    results = []
    for combination in random_combinations:
        params = {key: combination[i] for i, key in enumerate(params_grid.keys())}
        score = run_experiment(algorithm, params, data, k, max_iter)
        results.append((params, score))
    return results

def plot_and_save_sn_ratios(sn_ratios, params_grid, algorithm_name, output_dir):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Main Effects Plot for S/N Ratio ({algorithm_name})')

    for ax, (key, sn_values) in zip(axs.flatten(), sn_ratios.items()):
        ax.plot(params_grid[key], sn_values, marker='o')
        ax.set_title(key)
        ax.set_xlabel('Levels')
        ax.set_ylabel('S/N Ratio')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{algorithm_name}_sn_ratio_plot.png'))
    plt.close()

if __name__ == "__main__":
    random.seed(42)
    k = 15
    max_iter = 100

    dataset_paths = [
        './dataset/Input/20newsgroups_dataset1.csv',
        './dataset/Input/reuters.csv',
        './dataset/Input/dbpedia_dataset.csv'
    ]

    num_experiments = 10

    output_base_dir = './dataset/Output/'
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)

    pso_params_grid = {
        'num_particles': [20, 30, 40],
        'inertia_weight': [0.5, 0.7, 0.9],
        'c1': [1, 1.5, 2],
        'c2': [1, 1.5, 2]
    }

    ga_params_grid = {
        'population_size': [50, 100, 150],
        'num_generations': [50, 100, 150],
        'crossover_rate': [0.5, 0.6, 0.7],
        'mutation_rate': [0.1, 0.15, 0.2]
    }

    sa_params_grid = {
        'initial_temp': [100, 250, 750, 1000],
        'cooling_rate': [0.85, 0.9, 0.95, 0.99]
    }

    for dataset_path in dataset_paths:
        print(f'Processing dataset: {dataset_path}')

        df = pd.read_csv(dataset_path, encoding='utf-8')
        le = LabelEncoder()
        true_labels = le.fit_transform(df['label'])
        Y = spectral_embedding(df, k)

        dataset_name = os.path.basename(dataset_path).split('.')[0]
        output_dir = os.path.join(output_base_dir, dataset_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        pso_random_combinations = create_random_combinations(pso_params_grid, num_experiments)
        ga_random_combinations = create_random_combinations(ga_params_grid, num_experiments)
        sa_random_combinations = create_random_combinations(sa_params_grid, num_experiments)

        print('Running PSO experiment...')
        pso_results = run_taguchi_experiment('pso', pso_params_grid, Y, k, max_iter, pso_random_combinations)
        print('PSO complete')

        print('Running GA experiment...')
        ga_results = run_taguchi_experiment('ga', ga_params_grid, Y, k, max_iter, ga_random_combinations)
        print('GA complete')

        print('Running SA experiment...')
        sa_results = run_taguchi_experiment('sa', sa_params_grid, Y, k, max_iter, sa_random_combinations)
        print('SA complete')

        for algorithm_name, results, params_grid in zip(
                ['pso', 'ga', 'sa'], [pso_results, ga_results, sa_results],
                [pso_params_grid, ga_params_grid, sa_params_grid]
        ):
            scores = [result[1] for result in results]
            sn_ratio = calculate_sn_ratio_minimization(scores)

            sn_ratios = {key: [] for key in params_grid.keys()}
            for key in params_grid.keys():
                for level in params_grid[key]:
                    filtered_scores = [result[1] for result in results if result[0][key] == level]
                    sn_ratios[key].append(calculate_sn_ratio_minimization(filtered_scores))

            with open(os.path.join(output_dir, f'{algorithm_name}_results.txt'), 'w') as f:
                for result in results:
                    f.write(f"Params: {result[0]}, Score: {result[1]}\n")
                f.write(f"S/N Ratio: {sn_ratio}\n")

            plot_and_save_sn_ratios(sn_ratios, params_grid, algorithm_name, output_dir)

        print(f'Finished processing dataset: {dataset_path}')