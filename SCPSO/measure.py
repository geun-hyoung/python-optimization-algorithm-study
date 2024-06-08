import warnings
import numpy as np

warnings.filterwarnings('ignore')

from scipy.special import comb
from scipy.stats import entropy
from sklearn.metrics.cluster import contingency_matrix

def calculate_accuracy(true_labels, predicted_labels):
    # Contingency matrix
    contingency = contingency_matrix(true_labels, predicted_labels)

    # Initialize counts
    TP = TN = FP = FN = 0

    # Calculate TP, TN, FP, FN
    for i in range(len(true_labels)):
        for j in range(i + 1, len(true_labels)):
            same_cluster_pred = (predicted_labels[i] == predicted_labels[j])
            same_cluster_true = (true_labels[i] == true_labels[j])

            if same_cluster_true and same_cluster_pred:
                TP += 1
            elif not same_cluster_true and not same_cluster_pred:
                TN += 1
            elif not same_cluster_true and same_cluster_pred:
                FP += 1
            elif same_cluster_true and not same_cluster_pred:
                FN += 1

    # Accuracy calculation
    print("TP",TP, "TN", TN, "FP", FP, "FN", FN)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy

def compute_conditional_entropy(true_labels, cluster_labels):
    # 교차 테이블(Contingency Table) 생성
    cont_mat = contingency_matrix(true_labels, cluster_labels)
    # 클러스터별 총 개수
    cluster_sizes = cont_mat.sum(axis=0)
    # 전체 샘플 수
    total_samples = np.sum(cluster_sizes)

    # 각 클러스터에 대한 조건부 엔트로피 계산
    conditional_entropies = []
    for k in range(cont_mat.shape[1]):
        cluster_prob = cluster_sizes[k] / total_samples
        cluster_entropy = entropy(cont_mat[:, k])
        weighted_entropy = cluster_prob * cluster_entropy
        conditional_entropies.append(weighted_entropy)

    # 조건부 엔트로피의 합을 반환
    return np.sum(conditional_entropies)

def normalized_mutual_info_score_v2(true_labels, cluster_labels):
    """NMI를 계산하는 함수 (또 다른 정의)"""
    H_C = entropy(cluster_labels)
    H_K = entropy(true_labels)

    I_CK = compute_conditional_entropy(true_labels, cluster_labels)

    NMI = 2 * (H_C-I_CK) / (H_C + H_K)
    return NMI

def compute_comb(n, k):
    return comb(n, k, exact=True)

def compute_index_terms(cont_mat):
    n_ij = np.sum([compute_comb(nij, 2) for nij in cont_mat.flatten()])

    row_sums = np.sum(cont_mat, axis=1)
    n_i = np.sum([compute_comb(row_sum, 2) for row_sum in row_sums])

    col_sums = np.sum(cont_mat, axis=0)
    n_j = np.sum([compute_comb(col_sum, 2) for col_sum in col_sums])

    return n_ij, n_i, n_j

def compute_expected_index(b, c, total_combinations):
    return (b * c) / total_combinations

def compute_ari(true_labels, cluster_labels):
    """ARI (Adjusted Rand Index)를 계산합니다."""
    cont_mat = contingency_matrix(true_labels, cluster_labels)
    print(cont_mat)
    total_samples = np.sum(cont_mat)

    n_ij, n_i, n_j = compute_index_terms(cont_mat)

    expected_index = (n_i * n_j) / compute_comb(total_samples, 2)
    ari = (n_ij - expected_index) / ((0.5 * (n_i)) - expected_index)
    return ari