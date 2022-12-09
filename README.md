# Sentiment of News Articles
## Authors: Eva Engel & Tori Leatherman
Determining positive or negative sentiment of news articles.

Our goal is to create a ML pipeline that culminates in a UI that users can input a URL for some news article and receive whether the article has an overall negative or positive sentiment. An idea, pending the successful completion of our initial goal, would be to develop this further, allowing the crawling of a news site and returning a list of all articles and their associated sentiment. We hope the completion of this project will empower people to be more in control of the media they intake on a daily basis, adapting to their personalized preferences.

We will utilize the dataset found here: https://huggingface.co/datasets/ag_news for our training and evaluation datasets. Depending on the robustness of our model, we may add more data as we see fit. We intend to use a pretrained transformer to conduct NLP sentiment classification. We plan to utilize a feature pipeline, training pipeline, and batch inference pipeline so that our ML system is completely scalable. We will use HuggingFace to create a UI that allows users to screen any news article for its sentiment before reading. 
