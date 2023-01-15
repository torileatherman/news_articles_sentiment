import gradio as gr
from datasets import load_dataset, Dataset
import pandas as pd
from huggingface_hub import create_repo
from huggingface_hub import login
login()

# Load batch predictions data set
dataset = load_dataset("torileatherman/sentiment_analysis_batch_predictions", split='train')
predictions_df = pd.DataFrame(dataset)
grouped_predictions = predictions_df.groupby(predictions_df.Prediction)
positive_preds = grouped_predictions.get_group(2)
neutral_preds = grouped_predictions.get_group(1)
negative_preds = grouped_predictions.get_group(0)

predictions_df['Prediction'] = predictions_df['Prediction'].map({0: 'Negative', 1: 'Neutral', 2: 'Positive'})

# Load training data set
dataset = load_dataset("torileatherman/sentiment_analysis_training", split='train')
training_df = pd.DataFrame(dataset)
random_sample = {}

# Number of articles shown
n = 5

def article_selection(sentiment):
    if sentiment == "Positive":
        predictions = positive_preds
        predictions_shuffled = predictions.sample(frac=1,weights=predictions['Confidence'])
        top3 = predictions_shuffled[0:n]
        top3_result = top3[['Headline_string','Url']]
        top3_result.rename(columns = {'Headline_string':'Headlines', 'Url':'URL'})
        return top3_result
    
    elif sentiment == "Negative":
        predictions = negative_preds
        predictions_shuffled = predictions.sample(frac=1,weights=predictions['Confidence'])
        top3 = predictions_shuffled[0:n]
        top3_result = top3[['Headline_string','Url']]
        top3_result.rename(columns = {'Headline_string':'Headlines', 'Url':'URL'})
        return top3_result
    else:
        predictions = neutral_preds
        predictions_shuffled = predictions.sample(frac=1,weights=predictions['Confidence'])
        top3 = predictions_shuffled[0:n]
        top3_result = top3[['Headline_string','Url']]
        top3_result.rename(columns = {'Headline_string':'Headlines', 'Url':'URL'})
        return top3_result

def manual_label():
    # Selecting random row from batch data
    global random_sample
    random_sample = predictions_df.sample()
    random_headline = random_sample['Headline_string'].iloc[0]
    random_prediction = random_sample['Prediction'].iloc[0]
    
    return random_headline, random_prediction


def thanks(sentiment):

    # Create int label 
    mapping = gender = {'Negative': 0,'Neutral': 1, 'Positive':2}
    sentiment = int(mapping[sentiment])
    
    global training_df
    # Append training data set
    training_df = training_df.append({'Sentiment': sentiment, 'Headline_string': random_sample['Headline_string'].iloc[0], 'Headline': random_sample['Headline'].iloc[0] }, ignore_index=True)
    training_df = training_df.drop_duplicates(subset='Headline_string').reset_index(drop=True)
    
    # Upload training data set
    ds = Dataset.from_pandas(training_df)
    try: 
        ds.push_to_hub("torileatherman/sentiment_analysis_training")
    except StopIteration:
        pass

    return f"""Thank you for making our model better! """

description1 = "This application recommends news articles depending on the sentiment of the headline. Enter your preference of what type of news articles you would like recommended to you today: Positive, Negative, or Neutral."


suggestion_demo = gr.Interface(
    fn=article_selection,
    title = 'Recommending News Articles',
    inputs = gr.Dropdown(["Positive","Negative","Neutral"], label="What type of news articles would you like recommended?"),
    outputs = "dataframe",
    #outputs = [gr.Textbox(label="Recommended News Articles (1/3)"),gr.Textbox(label="Recommended News Articles (2/3)"),gr.Textbox(label="Recommended News Articles (3/3)")],
    description = "This application recommends news articles depending on the sentiment of the headline. Enter your preference of what type of news articles you would like recommended to you today: Positive, Negative, or Neutral."
)

with gr.Blocks() as manual_label_demo:
    gr.Markdown("<h1 style='text-align: center;'> Label our Data</h1> This application will show you a random news headline and our predicted sentiment. In order to improve our model, choose the real sentiment of this headline from our dropdown and submit!")
    generate_btn = gr.Button('Show me a headline!')
    generate_btn.click(fn=manual_label, outputs=[gr.Textbox(label="News Headline"),gr.Textbox(label="Our Predicted Sentiment")])
    drop_down_label = gr.Dropdown(["Positive","Negative","Neutral"], label="Select the true sentiment of the news article.")
    submit_btn =  gr.Button('Submit your sentiment!')
    submit_btn.click(fn=thanks, inputs=drop_down_label, outputs=gr.Textbox(label = ' '))


demo = gr.TabbedInterface([suggestion_demo, manual_label_demo], ["Get recommended news articles", "Help improve our model"])


demo.launch()
