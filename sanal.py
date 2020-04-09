
import pandas as pd

# Matplot
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

# nltk
import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer

# Utility
import re
import numpy as np
import os
from collections import Counter
import logging
import time
import pickle
import itertools

# Set log
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# In[2]:


from keras.layers import Input
from keras.models import Model

nltk.download('stopwords')


# In[65]:


# DATASET
DATASET_COLUMNS = ["target", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
TRAIN_SIZE = 0.8

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"


# KERAS
SEQUENCE_LENGTH = 300
EPOCHS = 8
BATCH_SIZE = 1024

# SENTIMENT
POSITIVE = "POSITIVE"
NEGATIVE = "NEGATIVE"
NEUTRAL = "NEUTRAL"
SENTIMENT_THRESHOLDS = (0.4, 0.7)

# EXPORT
KERAS_MODEL = "model.h5"
WORD2VEC_MODEL = "model.w2v"
TOKENIZER_MODEL = "tokenizer.pkl"
ENCODER_MODEL = "encoder.pkl"


# In[5]:


dataset_path = "sentiment_analysis/training.1600000.processed.noemoticon.csv"
df = pd.read_csv(dataset_path, encoding =DATASET_ENCODING , names=DATASET_COLUMNS)


# In[6]:


print("Dataset size:", len(df))


# In[7]:


stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")


# In[8]:


def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)


# In[9]:


get_ipython().run_cell_magic('time', '', 'df.text = df.text.apply(lambda x: preprocess(x))')


# In[10]:


def readglovevec(file_name):
    with open(file_name,encoding = 'utf8') as f:
        words = set()
        word_to_vec_map = {}
        con = f.readlines()
        for line in con:
            line = line.strip().split()
            c_word = line[0]
            words.add(c_word)
            word_to_vec_map[c_word] = np.array(line[1:],dtype = np.float64)
    i = 1
    word_to_index = {}
    index_to_word = {}
    for w in sorted(words):
        word_to_index[w] = i
        index_to_word[i] = w
        i = i+1
    return word_to_index,index_to_word,word_to_vec_map


# In[11]:


wi,iw,wv = readglovevec('sentiment_analysis/glove.6B.50d.txt')


# In[56]:


def tweet_to_indices(x,word_to_indices,max_len):
    m = len(x)
    X = np.zeros((m,max_len))
    
    for i in range(m):
        j = 0
        for w in x[i]:
            w = w.lower()
            if j>=max_len:
                break
            if w in wi.keys():
                X[i,j] = wi[w]
            j = j+1            
    return X
max_len = 10


# In[32]:


from sklearn import preprocessing


# In[33]:


def pretrained_embedding_layer(word_to_vec,word_to_index):
    vocab_len = len(word_to_index) + 1                  
    emb_dim = word_to_vec["apple"].shape[0]
    
    emb_matrix = np.zeros((vocab_len,emb_dim))
    
    for word,index in word_to_index.items():
        emb_matrix[index,:] = word_to_vec[word]
        
    embedding_layer = Embedding(vocab_len,emb_dim, trainable = False)
    
    embedding_layer.build((None,emb_dim))
    embedding_layer.set_weights([emb_matrix])
    
    return embedding_layer


# In[34]:


def convert_to_bin(x):
    if x == 4:
        x = 1
    return x


# In[35]:


df.target = df.target.apply(lambda x:convert_to_bin(x))


# In[36]:


Counter(df.target)


# In[66]:


df_train, df_test = train_test_split(df, test_size=1-TRAIN_SIZE, random_state=42)
print("TRAIN size:", len(df_train))
print("TEST size:", len(df_test))


# In[69]:


get_ipython().run_cell_magic('time', '', 'documents = [_text.split() for _text in df_train.text] ')


# In[70]:


ls = []
for i in range(len(documents)):
    ls.append(len(documents[i]))
print(max(ls))
print(np.mean(ls))
print(np.std(ls))


# In[71]:


tweet_indices = tweet_to_indices(documents,wi,max_len)


# In[72]:


print(Counter(df_train.target))
print(Counter(df_test.target))


# In[73]:


print(tweet_indices.shape)


# In[74]:


def model(input_shape, word_to_vec_map, word_to_index):
    tweet_indices = Input(shape = input_shape, dtype = 'int32')
    embedding_layer = pretrained_embedding_layer(word_to_vec_map,word_to_index)
    embeddings = embedding_layer(tweet_indices)
    X = Dropout(0.5)(embeddings)
    X = LSTM(100,dropout=0.2, recurrent_dropout=0.2)(X)
    X = Dense(1,activation = 'sigmoid')(X)
    
    model = Model(inputs = tweet_indices, outputs = X)
    
    return model


# In[75]:


model = model((max_len,),wv,wi)
model.summary()


# In[76]:


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[77]:


X = tweet_indices

#print(X.shape)
Y = df_train.target
print(X.shape,Y.shape)


# In[79]:


model.fit(X,Y,epochs  =2, batch_size = 128,shuffle = True, validation_split = 0.1)


# In[304]:


st = ["I am not a bad person","I am a good person"]
ls = [i.split() for i in st]
print(ls)


ind = tweet_to_indices(ls,wi,max_len)
print(ind)
print(wi['is'])


Y_pre = model.predict(ind)


print(Y_pre.shape[0])
print(Y_pre)
labe = []
for i in range(Y_pre.shape[0]):
    if Y_pre[i] < 0.3:
        labe.append('NEGATIVE')
    elif Y_pre[i] < 0.7:
        labe.append('NEUTRAL')
    else:
        labe.append('POSITIVE')

print(labe)



for i in range(len(labe)):
    print(st[i], 'depicts', labe[i], 'sentiment')




import tweepy

consumerKey = 'pqN5ROaL6jONqzcxstuTotKYk'
consumerSecret = 'e0HL0YhdzsNrYa5Mz8DuGDjDiqvTc6Z9384V71e7t8LIB3cNly'
accessToken = '910732003779186688-62gn4lcS3cr2U33z5F8PSF69wheWTPD'
accessTokenSecret = 'G4x69RFE1rdO9LWFgQUUCbquJzbDVcW9KBykI11wx7dAu'
auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth)

# input for term to be searched and how many tweets to search
search_term = "Odd Even"
nterms = 1000

# searching for tweets
tweets = tweepy.Cursor(api.search, q=searchTerm, lang = "en",tweet_mode = 'extended').items(NoOfTerms)

print(type(tweets))

text = []

for i in tweets:
    text.append(preprocess(''.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ",i.full_text))))


print(len(text))
tweets = [_text.split() for _text in text] 

print(len(tweets))
print(text[:5])
print(tweets[:5])

lst = []
for i in range(len(tweets)):
    lst.append(len(tweets[i]))
print(max(lst))
print(np.mean(lst))

tindices = tweet_to_indices(tweets,wi,10)

Y_pred = model.predict(tindices)
print(Y_pred.shape)

label = []
for i in range(Y_pred.shape[0]):
    if Y_pred[i] < 0.2:
        label.append('NEGATIVE')
    elif Y_pred[i] < 0.5:
        label.append('NEUTRAL')
    else:
        label.append('POSITIVE')
        
sentiment_dic = Counter(label)
print(sentiment_dic)
def plotPieChart(positive,negative, neutral,nterms,search_term):
        labels = ['Positive [' + str((sentiment_dic['POSITIVE']/nterms)*100) + '%]','Neutral [' + str((sentiment_dic['NEUTRAL']/nterms)*100) + '%]',
                  'Negative [' + str((sentiment_dic['NEGATIVE']/nterms)*100) + '%]']
        sizes = [positive, neutral, negative]
        colors = ['darkgreen', 'gold', 'red']
        patches, texts = plt.pie(sizes, colors=colors, startangle=90)
        plt.legend(patches, labels, loc="best")
        plt.title('How people are reacting on ' + search_term + ' by analyzing ' + str(nterms) + ' Tweets.')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()


plotPieChart(sentiment_dic['POSITIVE']/nterms,sentiment_dic['NEGATIVE']/nterms,sentiment_dic['NEUTRAL']/nterms,nterms,search_term)





