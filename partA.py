import pandas as pd
import numpy as np
import string
import nltk
import gensim
import pickle
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
nltk.download('punkt')
porter_stemmer = PorterStemmer()
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('english')


df = pd.read_csv('Precily_Text_Similarity.csv')
def preprocess(text):
    #Making the text punctuation free
    text ="".join([i for i in text if i not in string.punctuation])

    #Lowercasing the text
    text = text.lower()

    #Removing stopwords
    text = " ".join([i for i in text.split() if i not in stopwords])

    return text

#preprocessing the text in dataframe
df.text1 = df.text1.apply(lambda x: preprocess(x))
df.text2 = df.text2.apply(lambda x: preprocess(x))
df['text'] = df['text1'] + df['text2']


d1 = list(df.text1.values)
d2 = list(df.text2.values)
data = d1 + d2

#Making a model using doc2vec to get paragraph embeddings
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
max_epochs = 10
vec_size = 300
alpha = 0.025

model = Doc2Vec(vector_size=vec_size,
                alpha=alpha, 
                min_alpha= 0.0025,
                min_count=1,
                dm =1)
  
model.build_vocab(tagged_data)
model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=max_epochs)
model.save('doc2vec')

def get_similarity(text_dict):
    text1 =  text_dict['text1'] 
    text2 =  text_dict['text2']
    text1 = preprocess(text1)
    text2 = preprocess(text2)
    text1 =  word_tokenize(text1)
    text2 = word_tokenize(text2)
    model_test = Doc2Vec.load('doc2vec')
    sim = model_test.similarity_unseen_docs(text1,text2)
    return sim





