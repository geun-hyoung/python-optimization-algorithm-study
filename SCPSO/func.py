import re
import nltk
import warnings
import numpy as np

from nltk import word_tokenize
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from scipy.special import comb
from scipy.stats import entropy
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import adjusted_rand_score

nltk.download('stopwords')
stopWords = stopwords.words('english')
charfilter = re.compile('[a-zA-Z]+')

def simple_tokenizer(text):
    #tokenizing the words:
    words = word_tokenize(text)
    #converting all the tokens to lower case:
    words = map(lambda word: word.lower(), words)
    #let's remove every stopwords
    words = [word for word in words if word not in stopWords]
    #stemming all the tokens
    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
    ntokens = list(filter(lambda token : charfilter.match(token),tokens))
    return ntokens

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

def compute_contingency_matrix(true_labels, cluster_labels):
    """교차 테이블(Contingency Table)을 생성합니다."""
    return contingency_matrix(true_labels, cluster_labels)

def compute_comb(n, k):
    """조합(combination)을 계산합니다."""
    return comb(n, k, exact=True)

def compute_index_terms(cont_mat):
    """교차 테이블에서 필요한 값을 추출합니다."""
    print(cont_mat.flatten())
    n_ij = np.sum([compute_comb(nij, 2) for nij in cont_mat.flatten()])

    row_sums = np.sum(cont_mat, axis=1)
    n_i = np.sum([compute_comb(row_sum, 2) for row_sum in row_sums])

    col_sums = np.sum(cont_mat, axis=0)
    n_j = np.sum([compute_comb(col_sum, 2) for col_sum in col_sums])
    print(n_ij, n_i, n_j)
    return n_ij, n_i, n_j

def compute_expected_index(b, c, total_combinations):
    """기대 인덱스(E) 계산"""
    return (b * c) / total_combinations

def compute_ari(true_labels, cluster_labels):
    """ARI (Adjusted Rand Index)를 계산합니다."""
    cont_mat = compute_contingency_matrix(true_labels, cluster_labels)
    total_samples = np.sum(cont_mat)
    total_combinations = compute_comb(total_samples, 2)

    n_ij, n_i, n_j = compute_index_terms(cont_mat)

    expected_index = compute_expected_index(n_i, n_j, total_combinations)

    ari = (n_ij - expected_index) / (0.5 * (n_i) - expected_index)
    return ari