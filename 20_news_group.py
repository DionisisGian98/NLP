import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')  
nltk.download('wordnet')

from sklearn.datasets import fetch_20newsgroups
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk import  pos_tag_sents,word_tokenize
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from scipy import sparse
from random import randint

stopwords = set(stopwords.words('english'))
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

def twenty_newsgroup_to_csv(subset):
    newsgroups_train = fetch_20newsgroups(subset=subset, remove=('headers', 'footers', 'quotes'))

    df = pd.DataFrame([newsgroups_train.data, newsgroups_train.target.tolist()]).T
    df.columns = ['text', 'target']

    targets = pd.DataFrame( newsgroups_train.target_names)
    targets.columns=['title']

    out = pd.merge(df, targets, left_on='target', right_index=True)
    out['date'] = pd.to_datetime('now')
    out.to_csv('20_newsgroup_'+subset+'.csv')
    
twenty_newsgroup_to_csv('train')
twenty_newsgroup_to_csv('test')


df_train = pd.read_csv('20_newsgroup_train.csv')
df_test = pd.read_csv('20_newsgroup_test.csv')
df_test_title = df_test[['text']]

lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
  return [lemmatizer.lemmatize(word, pos="v") for word,add in text]
def word_count_total(text):
  return len(text)
def lemma_freq(text):
  return Counter(text)

def preprocess(df):
  lemmatized_sentence = []
  
  df['text'] = df['text'].str.lower()
  df['text'] = df['text'].apply(str)
  closed_tags = ['CD','CC','DT','EX','IN','LS','MD','PDT','POS','PRP','PRP$','RP','TO','UH','WDT','WP','WP$','WRB']
  #tokenize,remove stop words/closed tags and pos_tag the df
  df['text'].dropna(inplace=True)
  #df_train['text'].apply(word_tokenize)
  df['text'] = df.apply(lambda row: nltk.word_tokenize(row['text']), axis=1)
  df['text'] = df['text'].apply(lambda x: [item for item in x if item.lower() not in stopwords])
  df['text'] = df.apply(lambda row: nltk.pos_tag(row['text']), axis=1)
  df['text'] = df['text'].apply(lambda x: [i for i in x if i[0].isalpha()])
  df['text'] = df['text'].apply(lambda x: [item for item in x if item[0] not in closed_tags])
  df['text'] = df['text'].apply(lemmatize_text)
  #df['WordCount'] = df['text'].apply(word_count_total)
  #df['text'] = df['text'].apply(lemma_freq)
  return df


df_train = preprocess(df_train)
df_test = preprocess(df_test)

#tf-idf
v = TfidfVectorizer()
x = v.fit_transform(df_train['text'].values.astype('U'))

#df consisting of words/td-idf scores
df_tfidf = pd.DataFrame(data =x.toarray(), columns=v.get_feature_names())
df1 = df_tfidf.max().sort_values(ascending=False)
d = dict(df1)
sorted_d = dict(sorted(d.items(), key=lambda item: item[1],reverse=True))
sorted = pd.DataFrame.from_dict(sorted_d,orient='index')

#τα 8000 που θέλουμε
df1 = sorted.head(8000)

#θέλουμε να υπολογίσουμε το IDF για τις συλλογές κειμένων Α και Ε 
df_test = df_test[['text']]

#για να βγει το ίδιο idf
df_train_head = df_train.head(7532)
df_train_head = df_train_head.sample(frac=1)
df_test = df_test.sample(frac=1)

#υπολογιζουμε το tf-idf
v = TfidfVectorizer(vocabulary= S)
x = v.fit_transform(df_train_head['text'].values.astype('U'))
k = v.fit_transform(df_test['text'].values.astype('U'))

#τα κειμενα μας σε vectors
arr_train = x.toarray()
arr_test = k.toarray()
print(arr_train[1].shape)
print(arr_test[100].shape)

value1 = randint(0, 7533)
value2 = randint(0, 7533)

#θα συγκρινουμε 2 διανυσματα 
text = df_train_head['text'].tolist()
target = df_train_head['target'].tolist()

train_array = np.array(arr_train[value1] ,dtype=np.float32)
test_array = np.array(arr_test[value2] ,dtype=np.float32)
train_sparse= sparse.csr_matrix(train_array)
test_sparse=sparse.csr_matrix(test_array)

cos_res = cosine_similarity(train_sparse, test_sparse)
print('thats my cosine similarity result :', cos_res)

euc_res = euclidean_distances(train_sparse, test_sparse)
print('thats my eukleidian similarity result :', euc_res)
if(cos_res > 0.5):
  print('to test text anikei sto target',target[value1])
else:
  print('to test text den anikei sto target')
