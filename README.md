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
Financial Data Headlines           |  Covid Data Headlines
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/113507754/212494293-1fd83a33-2804-4297-a3ac-a9717a025abe.png)  |  ![image](https://user-images.githubusercontent.com/113507754/212494298-835a956a-e52c-479f-8e96-4aaf8e2bc3c5.png)

SEN R Data Headlines           |  SEN AMT Data Headlines
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/113507754/212494314-7a8b94b2-c3e9-4d47-ac52-9ee287800354.png)  |  ![image](https://user-images.githubusercontent.com/113507754/212494318-bc14eb59-7c11-4bcc-9e73-56589c2a472e.png)


After concatenating the datasets together, we then graphed the distribution of sentiments. As seen, the data is relatively balanced, with the majority of sentiments Negative. This, of course, is in line with our intuition, since news headlines often skew towards negative sentiments. The balance of the data allows us to use accuracy as our primary metric of model evaluation, which will be seen in the Training Pipeline section. 

![image](https://user-images.githubusercontent.com/113507754/212493829-6ae6f3aa-ebfa-474e-93b9-01b330d8e6e1.png)

Finally, we created wordclouds representing the most frequent word associated with each of the sentiments: negative, positive and neutral.

Negative             |  Positive             |  Neutral
:-------------------------:|:-------------------------:-------------------------:|
![image](https://user-images.githubusercontent.com/113507754/212494383-a4735059-065c-4e45-a2a8-0b6f46d2b458.png)
  |  ![image](https://user-images.githubusercontent.com/113507754/212494386-2d4a7781-325d-49c0-815e-c985ed773615.png)
  |  ![image](https://user-images.githubusercontent.com/113507754/212494393-5a8bba47-8612-4f15-96e6-12894f091321.png) 


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
