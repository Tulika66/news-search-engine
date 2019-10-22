

import numpy as np
import pandas as pd
import os
import glob
import nltk
import copy



from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from itertools import chain



tokenizer = RegexpTokenizer("\w+")
stemmer = nltk.PorterStemmer()


loc = os.getcwd()
term_freq = {}
class_freq = {}
file_class = {}
doc_Id = {}

empty_term_freq = {}
for i in range(1010):
    empty_term_freq[i] = 0

empty_class_freq = {}
empty_class_freq["business"] = 0
empty_class_freq["entertainment"] = 0
empty_class_freq["politics"] = 0
empty_class_freq["sport"] = 0
empty_class_freq["tech"] = 0
    
for i,file in enumerate(glob.glob(loc + "/bbc-fulltext/*/*/*.txt")):
    document = open(file).read()
    doc_Id[i] = file

    file_class[i] = file.split("/")[-2]
    tokenized = tokenizer.tokenize(document)
    for word in tokenized:
        stemmed = stemmer.stem(word)
        #if stemmed not in term_freq.keys():
        term_freq[stemmed] = copy.deepcopy(empty_term_freq)
        term_freq[stemmed][i] += 1
    
#remove stop words from dict
stop_word = set(stopwords.words("english"))
for word in stop_word:
    stemmed = stemmer.stem(word)
    if stemmed in term_freq.keys():
        del term_freq[stemmed]

tf = pd.DataFrame(term_freq)



#tf.to_csv("VSMTermFrequency.csv")
#cf.to_csv("VSMClassFrequency.csv")

#class_ = pd.DataFrame(file_class)
#x = pd.concat([tf, class_], axis = 1)


doc_freq = (tf > 0).sum(axis = 0)
inv_doc_freq = np.log10(1000/doc_freq)


tf = 1 + np.log10(tf)
tf.replace(to_replace = -np.inf, value = 0, inplace = True)



tf_idf = tf.multiply(inv_doc_freq, axis = 1)


business_doc_freq = np.log10(1 + ((tf.drop(index=range(200,1000)) > 0).sum(axis = 0)))
business_tf_idf_cf = (tf_idf.drop(index = (range(200,1000)))).multiply(business_doc_freq)


politics_doc_freq = np.log10(1 + ((tf.drop(index=chain(range(200), range(400,1000))) > 0).sum(axis = 0)))
politics_tf_idf_cf = (tf_idf.drop(index = chain(range(200), range(400,1000)))).multiply(politics_doc_freq)


entertainment_doc_freq = np.log10(1 + ((tf.drop(index=chain(range(400), range(600,1000))) > 0).sum(axis = 0)))
entertainment_tf_idf_cf = (tf_idf.drop(index = chain(range(400), range(600,1000)))).multiply(entertainment_doc_freq)


sport_doc_freq = np.log10(1 + ((tf.drop(index=chain(range(600), range(800,1000))) > 0).sum(axis = 0)))
sport_tf_idf_cf = (tf_idf.drop(index=chain(range(600), range(800,1000)))).multiply(sport_doc_freq)


# In[75]:


tech_doc_freq = np.log10(1 + ((tf.drop(index = range(800)) > 0).sum(axis = 0)))
tech_tf_idf_cf = (tf_idf.drop(index = range(800))).multiply(tech_doc_freq)


tf_idf_cf = business_tf_idf_cf.append([politics_tf_idf_cf, entertainment_tf_idf_cf,sport_tf_idf_cf, tech_tf_idf_cf])


norm_tf_idf_cf = tf_idf_cf.multiply( 1/ (((tf_idf_cf**2).sum(axis = 1))**0.5), axis= 0)

dict_word_index = {}
for i,word in enumerate(norm_tf_idf_cf.columns):
    dict_word_index[word] = i

empty_query_vector = np.zeros(14270)
query = input("Search")

tokenized = tokenizer.tokenize(query)
query_vector = copy.deepcopy(empty_query_vector)
for word in tokenized:
    stemmed = stemmer.stem(word)
    print(stemmed)
    if stemmed in dict_word_index.keys():
        #print(dict_word_index[stemmed])
        query_vector[dict_word_index[stemmed]] = 1
        #print(query_vector[dict_word_index[stemmed]])

x = {}
for i,row in enumerate(norm_tf_idf_cf.iterrows()):
    x[i] = np.matmul(row[1], query_vector.T)
x = sorted(x.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)


for i,doc in enumerate(x):
    file = doc_Id[doc[0]]
    document = open(file).read()
    print("Rank = " + str(i + 1))
   # print("DocId = " + str(doc) +" Category = " + file.split("/")[-2], end = "\t")
    #print(file)
    print(document)

    if i == 9:
        break




