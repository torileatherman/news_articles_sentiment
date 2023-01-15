import os
import re  # RegEx for removing non-letter characters

import hopsworks
import joblib
import nltk  # natural language processing
import pandas as pd
from datasets import Dataset, load_dataset
from hsml.model_schema import ModelSchema
from hsml.schema import Schema
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from newsapi.newsapi_client import NewsApiClient
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder

BACKFILL = False
ENCODER_EXIST = True

# Login to huggingface huggingface-cli login
# 1. Step: BACKFILL = True, ENCODER_EXIST = False
# 2. Step: BACKFILL = False, ENCODER_EXIST = True

# Parameter initialization -  arbitrarily chosen
voc_size = 5000 
max_len = 60

def headline_to_sequence(headline: str) -> str:
    ''' 
    This function converts headline text into a cleaned text that has been stemmed and stopwords removed

    :param headline: headline text from dataframe

    :return text: cleaned and processed headline text
    '''
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

def preprocess_inputs(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This function takes the inputed dataframe, drops any NA values, applies the headline_to_sequence
    function to the Headline column, and encodes the label of the Sentiment column. Returns dataframe

    :param df: unprocessed dataframe

    :return df: preprocessed dataframe
    '''
    df = df.copy()

    # drop missing rows
    df.dropna(axis=0, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['Headline_string'] = df['Headline']

    df['Headline'] = df['Headline'].apply(lambda x: headline_to_sequence(x)) 

    # Encode labels
    if 'Sentiment' in df.columns:
        le = LabelEncoder()
        df['Sentiment'] = le.fit_transform(df['Sentiment'])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print('The mapping of the labels: ', le_name_mapping)
    print('The data set is preprocessed.')
    return df

def tokenize_pad_sequences(df: pd.DataFrame) -> pd.DataFrame:
    '''
    This function tokenizes the input text into sequences of integers and then
    pad each sequence to the same length.

    :param df: preprocessed dataframe 
    :param max_len: length chosen for padding

    :return df: tokenized dataframe
    ''' 

    # Text tokenization
    tokenizer.fit_on_texts(df['Headline'])
    
    # Transforms text to a sequence of integers
    X = tokenizer.texts_to_sequences(df['Headline'])

    # Pad sequences to the same length
    X = pad_sequences(X, padding='post', maxlen=max_len)
    print('The data set is encoded and padded.')

    data = {"Headline": X.tolist()}
    df_tokenized = pd.DataFrame(data, columns=['Headline']) 

    df['Headline'] = df_tokenized['Headline']
    save_encoder_to_hw(input_schema = df['Headline'], output_schema = X)  

    return df

def save_encoder_to_hw(input_schema: Schema, output_schema: Schema) -> None:
    '''
    This function creates a model registry and model schema in Hopsworks,
    and saves the encoder to it.

    :param input_schema: input schema for the encoder model registry
    :param output_schema: output schema for the encoder model registry
    '''

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

def load_process() -> pd.DataFrame:
    '''
    This function reads the the training data from four sources, applies the preprocess_inputs and
    the tokenize_pad_sequences functions and returns a completely preprocessed dataframe.

    :return df: preprocessed dataframe ready for upload
    '''

    # Define the relative path to the data
    file_dir = os.path.dirname(__file__)   

    # Read all training data
    finance_data = pd.read_csv(os.path.join(file_dir, '../data/finance_data/all-data.csv'), names=['Sentiment','Headline'], encoding='latin-1')
    
    covid_data = pd.read_csv(os.path.join(file_dir, '../data/covid_data/raw_data.csv'), usecols = ['Sentiment','Headline'], encoding='latin-1')
    covid_data['Sentiment'] = covid_data['Sentiment'].map({0: 'negative', 1: 'positive'})

    sen_R_data = pd.read_csv(os.path.join(file_dir, '../data/sen_data/SEN_en_R.csv'), usecols = ['majority_label','headline'], encoding='latin-1')
    sen_R_data = sen_R_data.rename(columns = {"majority_label": "Sentiment", "headline": "Headline"})
    sen_R_data['Sentiment'] = sen_R_data['Sentiment'].map({'neg': 'negative', 'pos': 'positive', 'neutr': 'neutral'})   

    sen_AMT_data = pd.read_csv(os.path.join(file_dir, '../data/sen_data/SEN_en_AMT.csv'), usecols = ['majority_label','headline'], encoding='latin-1')
    sen_AMT_data = sen_AMT_data.rename(columns = {"majority_label": "Sentiment", "headline": "Headline"})
    sen_AMT_data['Sentiment'] = sen_AMT_data['Sentiment'].map({'neg': 'negative', 'pos': 'positive', 'neutr': 'neutral'})

    data = pd.concat([finance_data,covid_data,sen_R_data,sen_AMT_data], ignore_index=True)

    df_string = preprocess_inputs(data)
    df = tokenize_pad_sequences(df_string)

    return df

def scrape_process() -> pd.DataFrame:
    '''
    This functions calls the API to get recent batch data from five news sources, applies the 
    preprocess_inputs and the tokenize_pad_sequences functions and returns a completely preprocessed 
    dataframe.

    :return df: preprocessed dataframe ready for upload
    '''

    def get_headlines_url():
        top_headlines = list()
        for article in top_articles['articles']:
            top_headlines.append([article['title'], article['url']])

        return top_headlines

    api = NewsApiClient(api_key = 'ce3df7e49a0848f7ac670cac34703cfa') 
    data = pd.DataFrame()

    for source in ['business-insider', 'bbc-news', 'bloomberg', 'abc-news', 'cbs-news']:
        top_articles = api.get_top_headlines(sources = source) 

        top_headlines = get_headlines_url()
        articles_one_source = pd.DataFrame(top_headlines, columns=['Headline','Url'])
        data = pd.concat([data,articles_one_source]).drop_duplicates()

    df_string = preprocess_inputs(data)
    df = tokenize_pad_sequences(df_string)

    return df

if __name__ == "__main__":

    project = hopsworks.login()
    nltk.download("stopwords")

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
        # Read data and upload to huggingface
        headlines_sentiment_df = load_process()
        dataset = Dataset.from_pandas(headlines_sentiment_df)
        dataset.push_to_hub("eengel7/sentiment_analysis_training")

    else:
        # Load stored batch data set
        dataset = load_dataset("eengel7/sentiment_analysis_batch", split='train')
        batch_old = pd.DataFrame(dataset)

        # Scrape new headlines and check if they are in existing batch data
        headlines_scraped_df = scrape_process()
        batch_df = pd.concat([batch_old,headlines_scraped_df]).drop_duplicates(subset='Headline_string').reset_index(drop=True)

        # Upload new batch data set to huggingface
        dataset = Dataset.from_pandas(batch_df)
        dataset.push_to_hub("eengel7/sentiment_analysis_batch")