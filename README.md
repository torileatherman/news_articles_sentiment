# Sentiment of News Articles
## Authors: Eva Engel & Tori Leatherman

### Description
The focus of this project is creating an ML pipeline that predicts the sentiment of and recommends news articles based on their headlines. Our system uses the following pipelines and workflow:
1. Feature Pipeline to preprocess and store data in HuggingFace.
2. Training Pipeline to train our model and store it in Hopsworks.
3. Batch Inference Pipeline to predict sentiments on new batch data and store in HuggingFace.
4. HuggingFace App for recommending articles based on sentiment and allowing users to manually label our batch data to append to our training data.

We will explain each of these aspects in more detail in the following description.

### Data Source

TODO - EVA
#### Data-centric improvements
TODO - EVA

### Exploratory Data Analysis
Before preprocessing the data, we conducted an exploratory data analysis to understand the data from each individual source. The following images contain box plots of the headline length from the data sources containing financial data, covid data, and the two datasets from SEN.
Financial Data Headlines           |  Covid Data Headlines
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/113507754/212494293-1fd83a33-2804-4297-a3ac-a9717a025abe.png)  |  ![image](https://user-images.githubusercontent.com/113507754/212494298-835a956a-e52c-479f-8e96-4aaf8e2bc3c5.png)

SEN R Data Headlines           |  SEN AMT Data Headlines
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/113507754/212494314-7a8b94b2-c3e9-4d47-ac52-9ee287800354.png)  |  ![image](https://user-images.githubusercontent.com/113507754/212494318-bc14eb59-7c11-4bcc-9e73-56589c2a472e.png)


After concatenating the datasets together, we then graphed the distribution of sentiments. As seen, the data is relatively balanced, with the majority of sentiments Negative. This, of course, is in line with our intuition, since news headlines often skew towards negative sentiments. The balance of the data allows us to use accuracy as our primary metric of model evaluation, which will be seen in the Training Pipeline section. 

![image](https://user-images.githubusercontent.com/113507754/212493829-6ae6f3aa-ebfa-474e-93b9-01b330d8e6e1.png)

Finally, we created wordclouds representing the most frequent word associated with each of the sentiments: positive, negative and neutral.

Positive             |  Negative             |  Neutral
:-------------------------:|:-------------------------:|:-------------------------:|
![image](https://user-images.githubusercontent.com/113507754/212494383-a4735059-065c-4e45-a2a8-0b6f46d2b458.png) |  ![image](https://user-images.githubusercontent.com/113507754/212494386-2d4a7781-325d-49c0-815e-c985ed773615.png) |  ![image](https://user-images.githubusercontent.com/113507754/212494393-5a8bba47-8612-4f15-96e6-12894f091321.png) 


### Feature Pipeline

The feature pipeline is conducting three important functions: preprocessing initial training data from data sources, preprocessing batch data collecting using an API, and storing and updating the encoder used in the preprocessing steps. 

Using a parameter BACKFILL, if BACKFILL is set to True, we load and preprocess the initial training data from our various data sources. If BACKFILL is set to False, we call the API, NewsApiClient which scrapes the top headlines from a news source of our choice, in our case BBC. This is then also preprocessed. After either the training or batch data is preprocessed, it is pushed to HuggingFace to be stored. The other parameter of note is ENCODER_EXIST: if this is set to False, this will initialize the encoder and after preprocessing a dataset will be stored in the model registry on Hopsworks. If ENCODER_EXIST is set to true, then the encoder will be loaded from the model registry in Hopsworks, used for preprocessing and re-saved in Hopsworks with any updates experienced throught the preprocessing.

The steps to run the Feature Pipeline successfully are:
 1. Login to HuggingFace with authorization token
 2. Set BACKFILL = True, ENCODER_EXIST = False
 3. Set BACKFILL = False, ENCODER_EXIST = True
 
#### Preprocessing
We define four functions to perform preprocessing, and two functions to be utilized depending on the parameter BACKFILL as mentioned above. The first function defined, preprocess_inputs() receives the dataframe, and outputs a cleaned dataframe. The function drops any NA values and resets the indices, and creates a duplicate column 'Headline_string' which allows us to encode the column 'Headline' for our model. We then encode the column 'Sentiment' using LabelEncoder(). Additionally, we utilize a function, headline_to_sequence(), which receives a the column 'Headline' as a string, and outputs text that that has removed unusual characters, converted each letter to lowercase, tokenized and stemmed each word, and removed any stopwords using nltk.stopwords.

We then pass the dataframe through our function, tokenize_pad_sequences() which receives a dataframe, tokenizes the input text into sequences of integers, and pads each sequence to the same length. It will then return the processed dataframe. Within this function, after tokenizing the 'Headline', a function save_encoder_to_hw is called. This function first checks if there is an encoder already stored in Hopsworks, and if not will create the model and upload the recent version based on the encoding for the 'Headline' column.

These four functions are used in tandem, and then called by either the load_process() function if BACKFILL == True, or the scrape_process if BACKFILL == False. Finally when running this file, the preprocessed training data or preprocessed batch data will be pushed to HuggingFace for storage.

### Training Pipeline

#### Model 
TODO - EVA
#### Model-centric improvements
TODO - EVA


### Batch Inference Pipeline
The batch inference pipeline will first load the most recent model from Hopsworks. It will then load the most recent encoded batch data from HuggingFace, and predict the sentiment using the loaded model. After adding the predictions to the batch data, this will be pushed to HuggingFace to the batch_predictions dataset. This data will now be available for our HuggingFace app UI.

As mentioned in the Training Pipeline section, for every input, our model outputs a vector of three: the probability that the input is each of the sentiment values. In order to categorize the predictions definitively into one class, we use the argmax() function to take the index of the maximum value of this vector as the final prediction of our input. We then calculate the normalized confidence of each prediction in a new column 'Confidence' which we append to the batch data. This allows us to sort the batch_data depending on the confidence of the prediction, so we show headlines which the highest confidence. Thus after adding the 'Prediction' and 'Confidence' columns to our batch dataset, we upload this to our HuggingFace to be stored as the sentiment analysis batch predictions.

### HuggingFace App
The HuggingFace app created as an interactive UI can be found here: https://huggingface.co/spaces/torileatherman/news_headline_sentiment 

The UI has two tabs the user can interact with: 
1. The first tab is designed so the user can select if they would like to be recommended Positive, Negative, or Neutral headlines. Using our batch data stored in HuggingFace, we will then provide recent headlines that are categorized with the correct sentiment. 
2. The second tab is designed to allow users to manually label headlines from our recent batch. It will provide a recent news headline and our predicted sentiment; the user can then add their manual sentiment assessment. This labeled headline will be appended to our training data and re-saved in the HuggingFace dataset.
