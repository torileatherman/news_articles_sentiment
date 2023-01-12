import os
import re  # RegEx for removing non-letter characters

import nltk  # natural language processing
import numpy as np
import pandas as pd

nltk.download("stopwords")
import os

import hopsworks
import joblib
from hsml.model_schema import ModelSchema
from hsml.schema import Schema
from keras.preprocessing.text import Tokenizer, one_hot
from keras.utils import pad_sequences
from newsapi.newsapi_client import NewsApiClient
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder

BACKFILL = False
ENCODER_EXIST = True

# 1. Step: BACKFILL = True, ENCODER_EXIST = False
# 2. Step: BACKFILL = False, ENCODER_EXIST = True

# parameter initialization -  arbitrarily choosen
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

def tokenize_pad_sequences(df: pd.DataFrame):
    '''
    This function tokenize the input text into sequnences of intergers and then
    pad each sequence to the same length
    ''' 

    # Text tokenization
    tokenizer.fit_on_texts(df['Headline'])
    
    # Transforms text to a sequence of integers
    X = tokenizer.texts_to_sequences(df['Headline'])

    # Pad sequences to the same length
    X = pad_sequences(X, padding='post', maxlen=max_len)

    data = {"Headline": X.tolist()}
    df_tokenized = pd.DataFrame(data, columns=['Headline']) 

    df['Headline'] = df_tokenized['Headline']
    save_encoder_to_hw(input_schema = df['Headline'], output_schema = X)  

    return df

def save_encoder_to_hw(input_schema, output_schema):

    mr = project.get_model_registry()

    model_dir="headlines_sentiment_encoder"
    if os.path.isdir(model_dir) == False:
        os.mkdir(model_dir)

    joblib.dump(tokenizer, model_dir + "/headlines_sentiment_encoder.pkl")
    
    input_schema = Schema(input_schema)
    output_schema = Schema(output_schema)
    model_schema = ModelSchema(input_schema, output_schema)

    headlines_sentiment_encoder = mr.python.create_model(
        name = "headlines_sentiment_encoder", 
        model_schema=model_schema,
        description="Encoding strings"
    )
    headlines_sentiment_encoder.save(model_dir)

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

    df_string = preprocess_inputs(data)
    df = tokenize_pad_sequences(df_string)
    print('STATIC DATA \n', df)

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
    data = pd.DataFrame(top_headlines, columns=['Headline','Url'])
    
    df_string = preprocess_inputs(data)
    df = tokenize_pad_sequences(df_string)

    return df


#TODO
if __name__ == "__main__":
    project = hopsworks.login()
    fs = project.get_feature_store()

    if ENCODER_EXIST == True:
        # Load encoder from Hopsworks
        mr = project.get_model_registry()
        tokenizer = mr.get_model("headlines_sentiment_encoder", version=1)
        model_dir = tokenizer.download()
        tokenizer = joblib.load(model_dir + "/headlines_sentiment_encoder.pkl")
    else:
        # Initialise encoder
        tokenizer = Tokenizer(num_words=voc_size, lower=True, split=' ')

    if BACKFILL == True:
        headlines_sentiment_df = load_process()
        headlines_sentiment_fg = fs.get_or_create_feature_group(
            name = 'headlines_sentiment_fg',
            primary_key =['Headline','Sentiment'], 
            version = 1)
        
        headlines_sentiment_fg.insert(headlines_sentiment_df, write_options={"wait_for_job" : False})

    else:
        headlines_scraped_df = scrape_process()
        headlines_scraped_fg = fs.get_or_create_feature_group(
            name = 'headlines_scraped_fg',
            primary_key =['Headline','Url'], 
            version = 1)
        print('SCRAPED-----------')
        print(headlines_scraped_df)    
        headlines_scraped_fg.insert(headlines_scraped_df, write_options={"wait_for_job" : False})