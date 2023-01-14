# Sentiment of News Articles
## Authors: Eva Engel & Tori Leatherman
Determining positive or negative sentiment of news articles.

### Description
The focus of this project is creating an ML pipeline that predicts the sentiment of news articles based on their headlines. Our pipeline uses the following:
1. Huggingface to store feature group data.
2. Hopsworks to store the encoder and model.
3. Huggingface to create interactive UI.

We will explain each of these aspects in more detail in the following description.

### Data Source

TODO - EVA
#### Data-centric improvements
TODO - EVA

### Exploratory Data Analysis
Before preprocessing the data, we conducted an exploratory data analysis to understand the data from each individual source. The following images contain box plots of the headline length from the data sources containing financial data, covid data, and the two datasets from SEN.

![image](https://user-images.githubusercontent.com/113507754/212493156-19a2477f-d793-4226-a985-3934e380a77e.png)
![image](https://user-images.githubusercontent.com/113507754/212493136-34d38ef2-d555-469f-b0af-3397f27e012f.png)
![image](https://user-images.githubusercontent.com/113507754/212493205-614ef54d-bf99-431d-8876-3180a8c2c86b.png)
![image](https://user-images.githubusercontent.com/113507754/212493213-5e373312-2296-4d74-be1b-8d3ab47307a3.png)


### Feature Pipeline
The feature pipeline is conducting three important functions: preprocessing initial training data from data sources, preprocessing batch data collecting using an API, and storing and updating the encoder used in the preprocessing steps. 

Using a parameter BACKFILL, if BACKFILL is set to True, we load and preprocess the initial training data from our various data sources. If BACKFILL is set to False, we call the API, NewsApiClient which scrapes the top headlines from a news source of our choice, in our case BBC. This is then also preprocessed. After either the training or batch data is preprocessed, it is pushed to HuggingFace to be stored. The other parameter of note is ENCODER_EXIST: if this is set to False, this will initialize the encoder and after preprocessing a dataset will be stored in the model registry on Hopsworks. If ENCODER_EXIST is set to true, then the encoder will be loaded from the model registry in Hopsworks, used for preprocessing and re-saved in Hopsworks with any updates experienced throught the preprocessing.

The steps to run the Feature Pipeline successfully are:
 1. Login to HuggingFace with authorization token
 2. Set BACKFILL = True, ENCODER_EXIST = False
 3. Set BACKFILL = False, ENCODER_EXIST = True

### Training Pipeline

#### Model 
TODO - EVA
#### Model-centric improvements
TODO - EVA


### Batch Inference Pipeline
The batch inference pipeline will first load the most recent model from Hopsworks. It will then load the most recent encoded batch data from HuggingFace, and predict the sentiment using the loaded model. After adding the predictions to the batch data, this will be pushed to HuggingFace to the batch_predictions dataset. This data will now be available for our HuggingFace app UI.

### HuggingFace App
The HuggingFace app created as an interactive UI can be found here: https://huggingface.co/spaces/torileatherman/news_headline_sentiment As noted in the Batch Inference Pipeline, there are recent headlines and associated sentiment predictions stored in the batch_predictions dataset. This dataset will be loaded by the app for the interactive UI.

The first tab is designed so the user can select if they would like to be recommended Positive, Negative, or Neutral headlines. We will then provide recent headlines that are categorized with the correct sentiment. 
The second tab is designed to allow users to manually label headlines from our recent batch. It will provide a recent news headline and our predicted sentiment; the user can then add their manual sentiment assessment. This will update to our training data and improve our model.
