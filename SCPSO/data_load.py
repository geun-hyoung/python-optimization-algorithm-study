import os
import json
import urllib.request

import pandas as pd

import nltk
from nltk.corpus import reuters

from sklearn.datasets import fetch_20newsgroups

# reuters 데이터 셋
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

# 20newsgroups 데이터 셋
newsgroups_train = fetch_20newsgroups(subset='train')

train_data = newsgroups_train.data
train_labels = newsgroups_train.target
target_names = newsgroups_train.target_names

df_newsgroups = pd.DataFrame({'text': train_data, 'label': train_labels})
df_newsgroups['label'] = df_newsgroups['label'].apply(lambda x: target_names[x])
df_newsgroups.to_csv('./dataset/Input/20newsgroups_dataset.csv', index=False, encoding='utf-8')
print("20newsgroups 전체 데이터 개수:", len(df_newsgroups))

from datasets import load_dataset
import pandas as pd


dataset = load_dataset("dbpedia_14")

train_df = pd.DataFrame(dataset['train'])
category_names = dataset['train'].features['label'].names
train_df['label'] = train_df['label'].apply(lambda x: category_names[x])

train_df = train_df[['content', 'label']]
train_df.columns = ['text', 'label']

sample_size_per_category = 500
sampled_train_df = train_df.groupby('label').apply(lambda x: x.sample(sample_size_per_category)).reset_index(drop=True)
sampled_train_df.to_csv('./dataset/Input/dbpedia_dataset.csv', index=False, encoding='utf-8')
print("Train 데이터셋 크기:", sampled_train_df.shape)