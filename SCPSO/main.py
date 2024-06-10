import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from text_vectorization import spectral_embedding
from measure_metrics import evaluate_algorithms
from algorithms import pso, ga, SimulatedAnnealing, sk_means

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

    reuters_df = pd.read_csv('./dataset/Input/reuters_dataset.csv', encoding='utf-8')
    le = LabelEncoder()
    true_labels = le.fit_transform(reuters_df['label'])

    # spectral embedding
    Y = spectral_embedding(reuters_df, k)

    num_repeats = 10
    metrics = ['Accuracy', 'NMI', 'ARI']
    algorithm_type = ['pso', 'ga', 'sa', 'km']

    # 모든 결과를 저장할 데이터프레임 초기화
    results = {metric: pd.DataFrame(columns=['Iteration'] + [alg.upper() for alg in algorithm_type]) for metric in
               metrics}

    for algorithm in algorithm_type:
        for i in range(num_repeats):
            if algorithm == 'pso':
                labels, scores = pso(Y, k, num_particles, max_iter, inertia_weight, c1, c2)
            elif algorithm == 'ga':
                labels, scores = ga(Y, k, max_iter, population_size, crossover_rate, mutation_rate)
            elif algorithm == 'sa':
                labels, scores = SimulatedAnnealing(Y, k, initial_temp, cooling_rate, step_size, max_iter).run()
            elif algorithm == 'km':
                labels, scores = sk_means(Y, k)

            accuracy, nmi, ari= evaluate_algorithms(true_labels, labels)

            # 각 평가지표별로 결과를 저장
            results['Accuracy'] = results['Accuracy']._append({'Iteration': i + 1, algorithm.upper(): accuracy},ignore_index=True)
            results['NMI'] = results['NMI']._append({'Iteration': i + 1, algorithm.upper(): nmi}, ignore_index=True)
            results['ARI'] = results['ARI']._append({'Iteration': i + 1, algorithm.upper(): ari}, ignore_index=True)

    # 각 평가지표별로 평균값 추가
    for metric in metrics:
        mean_values = {alg.upper(): results[metric][alg.upper()].mean() for alg in algorithm_type}
        mean_values['Iteration'] = 'Mean'
        results[metric] = results[metric]._append(mean_values, ignore_index=True)

    # 엑셀 파일로 저장
    with pd.ExcelWriter('./dataset/algorithm_performance_metrics.xlsx') as writer:
        for metric in metrics:
            results[metric].to_excel(writer, sheet_name=metric, index=False)

    print("Done")