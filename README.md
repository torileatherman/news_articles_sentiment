# Sentiment of News Articles
Authors: Eva Engel & Tori Leatherman

## Description
The focus of this project is creating an ML pipeline that predicts the sentiment of and recommends news articles based on their headlines. Our system uses the following pipelines and workflow:
1. Feature Pipeline to preprocess and store data in HuggingFace.
2. Training Pipeline to train our model and store it in Hopsworks.
3. Batch Inference Pipeline to predict sentiments on new batch data and store in HuggingFace.
4. HuggingFace App for recommending articles based on sentiment and allowing users to manually label our batch data to append to our training data.

We will explain each of these aspects in more detail in the following description.

## Data Source
We initally identified two data sources via Kaggle to use, one containing [Covid-19 News headlines](https://www.kaggle.com/code/sameer1502/covid-19-news-sentiment-analysis/notebook), and the other containing [Financial News headlines](https://www.kaggle.com/datasets/ankurzing/sentiment-analysis-for-financial-news). Both these datasets also contain the associated sentiments of each headline. By the end of our data-centric improvements, we also added data from the paper [here](https://www.sciencedirect.com/science/article/pii/S1877050921018755).

### Data-centric improvements
When conducting our exploratory data analysis we noticed that only the Financial dataset contained neutral values along with positive and negative; whereas, the Covid dataset only contained positive and negative. Thus, we wanted to supplement the neutral data to enhance our model. In our search for additional data, we came across an research paper titled [Sentiment analysis of Entities in News headline](https://www.sciencedirect.com/science/article/pii/S1877050921018755). This paper consisted of presenting a novel publicly available human-labelled dataset of news headlines. As this was precisely what we were searching for, we requested access of the SEN R and SEN AMT datasets, which are the english sets for Researchers (R) and Amazon Mechanical Turk (AMT) services respectively. After gaining access we added these datasets to our training data, and as you will see in the below Exploratory Data Analysis section, they contributed to how balanced our training data became.

## Exploratory Data Analysis
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


## Feature Pipeline

The feature pipeline is conducting three important functions: preprocessing initial training data from data sources, preprocessing batch data collected using an API, and storing and updating the encoder used in the preprocessing steps. 

### Structure of Pipeline
There are two parameters, BACKFILL and ENCODER_EXIST, that control the flow of the pipeline. Using a parameter BACKFILL, if BACKFILL is set to True, we load and preprocess the initial training data from our various data sources. If BACKFILL is set to False, we call the API, NewsApiClient which scrapes the top headlines from a news source of our choice, in our case BBC. This is then also preprocessed. After either the training or batch data is preprocessed, it is pushed to HuggingFace to be stored. The other parameter of note is ENCODER_EXIST: if this is set to False, this will initialize the encoder and after preprocessing a dataset will be stored in the model registry on Hopsworks. If ENCODER_EXIST is set to true, then the encoder will be loaded from the model registry in Hopsworks, used for preprocessing and re-saved in Hopsworks with any updates experienced throught the preprocessing.

The steps to run the Feature Pipeline successfully are:
 1. Login to HuggingFace with authorization token
 2. Set BACKFILL = True, ENCODER_EXIST = False
 3. Set BACKFILL = False, ENCODER_EXIST = True
 
### Preprocessing
We define four functions to perform preprocessing, and two functions to be utilized depending on the parameter BACKFILL as mentioned above. The first function defined, preprocess_inputs() receives the dataframe, and outputs a cleaned dataframe. The function drops any NA values and resets the indices, and creates a duplicate column 'Headline_string' which allows us to encode the column 'Headline' for our model. We then encode the column 'Sentiment' using LabelEncoder(). Additionally, we utilize a function, headline_to_sequence(), which receives a the column 'Headline' as a string, and outputs text that that has removed unusual characters, converted each letter to lowercase, tokenized and stemmed each word, and removed any stopwords using nltk.stopwords.

We then pass the dataframe through our function, tokenize_pad_sequences() which receives a dataframe, tokenizes the input text into sequences of integers, and pads each sequence to the same length. It will then return the processed dataframe. Within this function, after tokenizing the 'Headline', a function save_encoder_to_hw is called. This function first checks if there is an encoder already stored in Hopsworks, and if not will create the model and upload the recent version based on the encoding for the 'Headline' column.

These four functions are used in tandem, and then called by either the load_process() function if BACKFILL == True, or the scrape_process if BACKFILL == False. Finally when running this file, the preprocessed training data or preprocessed batch data will be pushed to HuggingFace for storage.

## Training Pipeline
In this pipeline, we load the preprocessed training data from HuggingFace. We then split the training data and train the model. At the end, we upload the model to Hopsworks. 

### Model 
For the recurrent neural network, we used the Keras library and created a sequential model.  We based the underlying structure of the model on the project [AG News Classification LSTM](https://www.kaggle.com/code/ishandutta/ag-news-classification-lstm) although they solved a different classification task.  The first embedding layer takes the embedded and padded data and transforms them into dense vectors of a fixed size (here 40-dim). Followed by this, we add two *bidirectional LSTM* layers.  

#### Model architecture
The first embedding layer takes the embedded and padded data and transforms them into dense vectors of a fixed size (here 40-dim). Followed by this, we add two *bidirectional LSTM* layers.  
LSTM

A Long Short Term Memory Network(LSTM) addresses the problem of Vanishing and Exploding Gradients in a deep Recurrent Neural Network. An LSTM recurrent unit tries to “remember” all the past knowledge that the network is seen so far and to “forget” irrelevant data. This is done by introducing different activation function layers called “gates” for different purposes. Each LSTM recurrent unit also maintains a vector called the Internal Cell State which conceptually describes the information that was chosen to be retained by the previous LSTM recurrent unit. A Long Short Term Memory Network consists of four different gates for different purposes as described below:-
    1. Forget Gate(f): It determines to what extent to forget the previous data.
    2. Input Gate(i): It determines the extent of information to be written onto the Internal Cell State.
    3. Input Modulation Gate(g): It is often considered as a sub-part of the input gate and many literatures on LSTM’s do not even mention it and assume it inside the Input gate. It is used to modulate the information that the Input gate will write onto the Internal State Cell by adding non-linearity to the information and making the information Zero-mean. This is done to reduce the learning time as Zero-mean input has faster convergence. Although this gate’s actions are less important than the others and is often treated as a finesse-providing concept, it is good practice to include this gate into the structure of the LSTM unit.
    4. Output Gate(o): It determines what output(next Hidden State) to generate from the current Internal Cell State.
The basic work-flow of a Long Short Term Memory Network is similar to the work-flow of a Recurrent Neural Network with only difference being that the Internal Cell State is also passed forward along with the Hidden State.
BiDirectional LSTM -¶
Using bidirectional will run our inputs in two ways, one from past to future and one from future to past and what differs this approach from unidirectional is that in the LSTM that runs backwards we preserve information from the future and using the two hidden states combined we are able in any point in time to preserve information from both past and future.

After the two bidirectional LSTM layers, we added a Pooling Layer that decreases sensitivity to features, thereby creating more generalised data for better test results. This is followed by several combinations of Dense and Dropout Layers that end with an output layer with the softmax activiation function, and size 3 for the labels positive, negative and neutral. 


#### Model training
During training, we focused on the accuracy of the predicitions since the training data set is balanced. Since our classes are mutually exclusive we utilise sparse_categorical_crossentropy as a loss function (one can use categorical crossentropy when one sample can have multiple classes or labels are soft probabilities). We used the adam optimizer and  included EarlyStopping to stop at the epoch where val_accuracy does not improve significantly.
We make use of Wandb to monitor the accuracy of our training epochs and obtain the following reports for 20 epochs, with early stopping and a batch size of 256: 

Validation Accuracy           |  Accuracy
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/113507754/212552413-feda08ad-99f9-4c1e-b54d-bd6f41c9fdaf.png)  |  ![image](https://user-images.githubusercontent.com/113507754/212554647-f560210f-1821-4a0a-91eb-d17b2152778a.png)




### Model-centric improvements
As we can see from the diagrams included above, our model appears to be overfitting. This can be seen from the from the validation accuracy diagram which shows that after 3 epochs the validation accuracy reaches a maximum value and continues to decrease after that. To mitigate the effects of overfitting we tune the learning rate. Our first model has the default learning rate of 0.01. User a keras tuner, we perform a parameter search over the values [0.01, 0.001, 0.0001] with the objective of maximizing the validation accuracy. The optimal learning rate was identified as 0.001, which we retrained our model with. The diagrams obtained from Wandb for our training epochs can be found below: 
Validation Accuracy           |  Accuracy
:-------------------------:|:-------------------------:
![image](https://user-images.githubusercontent.com/113507754/212554488-2346512e-0e9e-4ac3-b11e-644c5dddb9e6.png)  |  ![image](https://user-images.githubusercontent.com/113507754/212554624-e3538a43-91cf-4602-b7cf-bf053d8a8804.png)

As seen in the above diagrams, at the end of the training epochs we have a higher score for both the validation accuracy and accuracy using the new learning rate.

## Batch Inference Pipeline
The batch inference pipeline will first load the most recent model from Hopsworks. It will then load the most recent encoded batch data from HuggingFace, and predict the sentiment using the loaded model. After adding the predictions to the batch data, the data will be sorted based on the confidence of the prediction. This data will then be pushed to HuggingFace to the batch_predictions dataset, and will be available for our HuggingFace app UI.

As mentioned in the Training Pipeline section, for every input, our model outputs a vector of three: the probability that the input is each of the sentiment values. In order to categorize the predictions definitively into one class, we use the argmax() function to take the index of the maximum value of this vector as the final prediction of our input. We then calculate the normalized confidence of each prediction in a new column 'Confidence' which we append to the batch data. This allows us to sort the batch_data depending on the confidence of the prediction, so we show headlines which the highest confidence. Thus after adding the 'Prediction' and 'Confidence' columns to our batch dataset, we upload this to our HuggingFace to be stored as the sentiment analysis batch predictions.

## HuggingFace App
The HuggingFace interactive UI can be found here: https://huggingface.co/spaces/torileatherman/news_headline_sentiment 

The UI has two tabs the user can interact with: 
1. The first tab is designed so the user can select if they would like to be recommended Positive, Negative, or Neutral headlines. Using our batch data stored in HuggingFace, we will then provide recent headlines that are categorized with the selected sentiment. 
2. The second tab is designed to allow users to manually label headlines from our recent batch. It will provide a recent news headline and our predicted sentiment; the user can then add their manual sentiment assessment. This manually labeled headline will be appended to our training data and re-saved in the HuggingFace dataset for future training purposes.
