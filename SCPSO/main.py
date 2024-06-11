import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from text_vectorization import spectral_embedding
from measure_metrics import evaluate_algorithms
from algorithms import pso, ga, SimulatedAnnealing

if __name__ == "__main__":
    # global param
    k = 15  # number_of_clusters
    max_iter = 100  # algorithms_iterations

    # pso params
    num_particles = 30
    inertia_weight = 0.5
    c1 = 1
    c2 = 1

    # ga params
    population_size = 100
    crossover_rate = 0.5
    mutation_rate = 0.1

    # aco params
    initial_temp = 1000
    cooling_rate = 0.85
    step_size = 0.01

    df = pd.read_csv('./dataset/Input/20newsgroups_dataset.csv', encoding='utf-8')
    le = LabelEncoder()
    true_labels = le.fit_transform(df['label'])

    # spectral embedding
    Y = spectral_embedding(df, k)

    num_repeats = 10
    metrics = ['Accuracy', 'NMI', 'ARI']
    algorithm_type = ['pso', 'ga', 'sa']

    # 모든 결과를 저장할 데이터프레임 초기화
    results = {metric: pd.DataFrame(columns=['Iteration'] + [alg.upper() for alg in algorithm_type]) for metric in metrics}
    rpd_results = pd.DataFrame(columns=['Iteration'] + [alg.upper() for alg in algorithm_type])

    best_fitness = float('inf')
    fitness_dict = {alg: [] for alg in algorithm_type}

    for algorithm in algorithm_type:
        for i in range(num_repeats):
            if algorithm == 'pso':
                labels, scores, fitness = pso(Y, k, num_particles, max_iter, inertia_weight, c1, c2)
            elif algorithm == 'ga':
                labels, scores, fitness = ga(Y, k, max_iter, population_size, crossover_rate, mutation_rate)
            elif algorithm == 'sa':
                labels, scores, fitness = SimulatedAnnealing(Y, k, initial_temp, cooling_rate, step_size, max_iter).run()

            accuracy, nmi, ari = evaluate_algorithms(true_labels, labels)

            # 각 평가지표별로 결과를 저장
            results['Accuracy'] = results['Accuracy']._append({'Iteration': i + 1, algorithm.upper(): accuracy}, ignore_index=True)
            results['NMI'] = results['NMI']._append({'Iteration': i + 1, algorithm.upper(): nmi}, ignore_index=True)
            results['ARI'] = results['ARI']._append({'Iteration': i + 1, algorithm.upper(): ari}, ignore_index=True)

            # 최적의 fitness 값 업데이트 및 저장
            fitness_dict[algorithm].append(fitness)
            if fitness < best_fitness:
                best_fitness = fitness

    # RPD 값을 계산하여 저장
    for algorithm in algorithm_type:
        for i in range(num_repeats):
            fitness = fitness_dict[algorithm][i]
            rpd = ((fitness - best_fitness) / best_fitness) * 100
            rpd_results = rpd_results._append({'Iteration': i + 1, algorithm.upper(): rpd}, ignore_index=True)

    # 각 평가지표별로 평균값 추가
    for metric in metrics:
        mean_values = {alg.upper(): results[metric][alg.upper()].mean() for alg in algorithm_type}
        mean_values['Iteration'] = 'Mean'
        results[metric] = results[metric]._append(mean_values, ignore_index=True)

    # RPD 평균값 추가
    mean_rpd_values = {alg.upper(): rpd_results[alg.upper()].mean() for alg in algorithm_type}
    mean_rpd_values['Iteration'] = 'Mean'
    rpd_results = rpd_results._append(mean_rpd_values, ignore_index=True)

    # 엑셀 파일로 저장
    with pd.ExcelWriter('./dataset/algorithm_performance_metrics.xlsx') as writer:
        for metric in metrics:
            results[metric].to_excel(writer, sheet_name=metric, index=False)
        rpd_results.to_excel(writer, sheet_name='RPD', index=False)

    # 기존 RPD Mean Plot 그리기
    plt.figure(figsize=(10, 6))
    for alg in algorithm_type:
        plt.plot(rpd_results[rpd_results['Iteration'] != 'Mean']['Iteration'],
                 rpd_results[rpd_results['Iteration'] != 'Mean'][alg.upper()],
                 label=alg.upper())
    plt.xlabel('Iteration')
    plt.ylabel('RPD (%)')
    plt.title('RPD Mean Plot')
    plt.legend()
    plt.grid(True)
    plt.savefig('./dataset/rpd_mean_plot.png')
    plt.show()

    # 각 알고리즘의 RPD 평균과 범위를 나타내는 선과 점 그리기
    mean_rpd = [rpd_results[alg.upper()].mean() for alg in algorithm_type]
    min_rpd = [rpd_results[alg.upper()].min() for alg in algorithm_type]
    max_rpd = [rpd_results[alg.upper()].max() for alg in algorithm_type]

    plt.figure(figsize=(10, 6))
    for i, alg in enumerate(algorithm_type):
        plt.errorbar(i, mean_rpd[i], yerr=[[mean_rpd[i] - min_rpd[i]], [max_rpd[i] - mean_rpd[i]]], fmt='o', capsize=5, color='skyblue', label=alg.upper())

    plt.xticks(range(len(algorithm_type)), algorithm_type)
    plt.xlabel('Algorithm')
    plt.ylabel('RPD (%)')
    plt.title('RPD Mean and Range for Each Algorithm')
    plt.savefig('./dataset/rpd_mean_range_plot.png')
    plt.show()

    print("Done")