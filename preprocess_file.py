# imports!
import sqlite3
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from nltk import  pos_tag_sents,word_tokenize
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import nltk
from collections import Counter
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

#Connect to the database
con = sqlite3.connect("save_pandas.db")
sql_query = ''' SELECT * FROM articles_info '''
df = pd.read_sql(sql_query, con)

# Preprocess Data 
lemmatizer = WordNetLemmatizer()
lemmatized_sentence = []
df['Article'] = df['Article'].str.lower()
closed_tags = ['CD','CC','DT','EX','IN','LS','MD','PDT','POS','PRP','PRP$','RP','TO','UH','WDT','WP','WP$','WRB','.',',',':','’','-']
#tokenize,remove stop words/closed tags and pos_tag the df
df['Article'] = df.apply(lambda row: nltk.word_tokenize(row['Article']), axis=1)
stopwords = set(stopwords.words('english'))
df['Article']=df['Article'].apply(lambda x: [item for item in x if item.lower() not in stopwords])
df['Article'] = df.apply(lambda row: nltk.pos_tag(row['Article']), axis=1)
df['Article'] = df['Article'].apply(lambda x: [i for i in x if i[0].isalpha()])
#print(df)

closed_tags = ['CD','CC','DT','EX','IN','LS','MD','PDT','POS','PRP','PRP$','RP','TO','UH','WDT','WP','WP$','WRB']
df['Article']=df['Article'].apply(lambda x: [item for item in x if item[1] not in closed_tags])

def lemmatize_text(text):
    return [lemmatizer.lemmatize(word, pos="v") for word,label in text]

def word_count_total(text):
    return len(text)

lemmatizer = nltk.stem.WordNetLemmatizer()
df['Article'] = df['Article'].apply(lemmatize_text)

df['WordCount'] = df['Article'].apply(word_count_total)
print(df['Article'])
#frequency of every
def lemma_freq(text):
    return Counter(text)

df['Article1'] = df['Article'].apply(lemma_freq)
print(df['Article1'])

def join(text):
    return " ".join(text)
df['Article'] = df['Article'].apply(join)

# Calculate tf-idf 
def tf_idf(text):
    dictionary = []
    word_count = len(text)
    text = dict(text)
    for word,count in text.items():
        sc = df[df['Article'].str.contains(word)]
        tf_idf = (count / word_count) * (df.shape[0]/len(sc))
        dictionary.append({word: tf_idf})        
    return dictionary 

#euretirio 
df['Article1'] = df['Article1'].apply(tf_idf)

# dict with lemmas and weights 
articles = df.Article1.tolist()
flat_list = [item for sublist in articles for item in sublist]
dictionary = dict()
for item in flat_list:
    for word,score in item.items():
        if word in dictionary:
            dictionary[word].append(score)
        else:
            dictionary[word] = [score]            
#dict to df
dict_df = pd.DataFrame.from_dict(dictionary, orient='index')
transpose = dict_df.transpose()
print(dict_df)
#dict to xml
f = open("dictionary.xml", "w")
f.write("<inverted_index>")
f.write("\n")
f.close()
for word,score in dictionary.items():
    f = open("dictionary.xml", "a")
    f.write('<lemma name="{}">'.format(word))
    f.write("\n")
    for i in score:
        f.write('<TF-IDF weight="{}"/>'.format(i))
        f.write("\n")
    f.write('   </lemma>')
    f.write("\n")
    f.close()
f = open("dictionary.xml", "a")
f.write("</inverted_index>")
f.write("\n")
f.close()
