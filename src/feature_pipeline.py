import numpy as np
import pandas as pd 
import os

#For Preprocessing
import re    # RegEx for removing non-letter characters
import nltk  #natural language processing
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem  import PorterStemmer

from sklearn.preprocessing import LabelEncoder 
import sklearn.preprocessing as pr
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras.preprocessing.text import Tokenizer, one_hot
from keras.utils import pad_sequences

#For data visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#matplotlib inline

import hopsworks
from newsapi.newsapi_client import NewsApiClient

LOCAL = True
BACKFILL = False

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name(modal_secret_name))
   def f():
       g()

#TODO
def g():
    project = hopsworks.login()
    fs = project.get_feature_store()

    if BACKFILL == True:
        headlines_sentiment_df = load_process()
        headlines_sentiment_fg = fs.get_or_create_feature_group(
            name = 'headlines_sentiment_fg',
            primary_key =['Headline','Sentiment'], 
            version = 3)
        headlines_sentiment_fg.insert(headlines_sentiment_df, write_options={"wait_for_job" : False})

    else:
        headlines_scraped_df = scrape_process()
        headlines_scraped_fg = fs.get_or_create_feature_group(
            name = 'headlines_scraped_fg',
            primary_key =['Headline','URL'], 
            version = 1)
            
        headlines_scraped_fg.insert(headlines_scraped_df, write_options={"wait_for_job" : False})

    
# pararmeter initialization
voc_size = 5000 
max_len = 60

def headline_to_sequence(headline):
    ''' Convert headline text into a sequence of words '''
    ps = PorterStemmer()

    text = re.sub(r'[^a-zA-Z0-9]',' ', headline)

    # convert to lowercase
    text = text.lower()

    # tokenize
    text = text.split()

    #remove stopwords and apply stemming
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
   
    return text

def preprocess_inputs(df):
    df = df.copy()

    # drop missing rows
    df.dropna(axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)


    df['Headline'] = df['Headline'].apply(lambda x: headline_to_sequence(x)) 

    # Encode labels
    if 'Sentiment' in df.columns:
        le = LabelEncoder()
        df['Sentiment'] = le.fit_transform(df['Sentiment'])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print('The mapping of the labels: ', le_name_mapping)
    
    return df

def encoder(data):
    df_string = preprocess_inputs(data)
  #arbitrarily choosen
    df_int = df_string.copy()
    df_int['Headline'] = df_string['Headline'].apply(lambda x: one_hot(x,voc_size))

    # Pad sequences to the same length  
    df = df_int.copy()

    df['Headline'] = df_int['Headline'].apply(lambda x: pad_sequences([x], padding='post', maxlen = max_len)[0])

    return df

def load_process():
    # Read all training data
    finance_data = pd.read_csv('../data/finance_data/all-data.csv', names=['Sentiment','Headline'], encoding='latin-1')
    
    covid_data = pd.read_csv('../data/covid_data/raw_data.csv', usecols = ['Sentiment','Headline'], encoding='latin-1')
    covid_data['Sentiment'] = covid_data['Sentiment'].map({0: 'negative', 1: 'positive'})

    sen_R_data = pd.read_csv('../data/sen_data/SEN_en_R.csv', usecols = ['majority_label','headline'], encoding='latin-1')
    sen_R_data = sen_R_data.rename(columns = {"majority_label": "Sentiment", "headline": "Headline"})
    sen_R_data['Sentiment'] = sen_R_data['Sentiment'].map({'neg': 'negative', 'pos': 'positive', 'neutr': 'neutral'})   

    sen_AMT_data = pd.read_csv('../data/sen_data/SEN_en_AMT.csv', usecols = ['majority_label','headline'], encoding='latin-1')
    sen_AMT_data = sen_AMT_data.rename(columns = {"majority_label": "Sentiment", "headline": "Headline"})
    sen_AMT_data['Sentiment'] = sen_AMT_data['Sentiment'].map({'neg': 'negative', 'pos': 'positive', 'neutr': 'neutral'})

    data = pd.concat([finance_data,covid_data,sen_R_data,sen_AMT_data], ignore_index=True)

    df = encoder(data)

    return df

def scrape_process():
    api = NewsApiClient(api_key = 'ce3df7e49a0848f7ac670cac34703cfa')  # TODO delete later
    top_articles = api.get_top_headlines(sources='bbc-news')

    def get_headlines_url():
        top_headlines = list()
        for article in top_articles['articles']:
            top_headlines.append([article['title'], article['url'] ])

        return top_headlines

    top_headlines = get_headlines_url()
    df = pd.DataFrame(top_headlines, columns=['Headline','Url'])
    
    df = encoder(df)

    return df
    

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()

