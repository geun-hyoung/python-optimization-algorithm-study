import nltk
from nltk.corpus import reuters
from sklearn.datasets import fetch_20newsgroups

import pandas as pd

# dataset 1
nltk.download('reuters')
files = reuters.fileids()

train_data = []
train_labels = []
for file in reuters.fileids():
        train_data.append(reuters.raw(file))
        train_labels.append(reuters.categories(file))

our_labels = reuters.categories()

docs_train = []
docs_labels = []
for i in range(len(train_labels)):
    if any(item in train_labels[i] for item in our_labels):
        docs_train.append(train_data[i])
        docs_labels.append(train_labels[i])

df_reuters = pd.DataFrame({'text': docs_train[:100], 'label': docs_labels[:100]})
df_reuters['label'] = df_reuters['label'].apply(lambda x: x[0])
df_reuters.to_csv('./dataset/Input/reuters_dataset.csv', index=False, encoding='utf-8')

print("reuters 전체 데이터 개수:", len(df_reuters))

# 20newsgroups 데이터셋 로드
newsgroups_train = fetch_20newsgroups(subset='train')

train_data = newsgroups_train.data
train_labels = newsgroups_train.target
target_names = newsgroups_train.target_names

df_newsgroups = pd.DataFrame({'text': train_data, 'label': train_labels})
df_newsgroups['label'] = df_newsgroups['label'].apply(lambda x: target_names[x])
df_newsgroups.to_csv('./dataset/Input/20newsgroups_dataset.csv', index=False, encoding='utf-8')
print("20newsgroups 전체 데이터 개수:", len(df_newsgroups))