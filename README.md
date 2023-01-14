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

### Feature Pipeline

### Training Pipeline

### Batch Inference Pipeline

### Huggingface App
The HuggingFace app created as an interactive UI can be found here: https://huggingface.co/spaces/torileatherman/news_headline_sentiment

The first tab is designed so the user can select if they would like to be recommended Positive, Negative, or Neutral headlines. We will then provide recent headlines that are categorized with the correct sentiment. The second tab is designed to allow users to manually label headlines from our recent batch. It will provide a recent news headline and our predicted sentiment; the user can then add their manual sentiment assessment. This will update to our training data and improve our model.
