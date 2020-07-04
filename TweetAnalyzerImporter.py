# %%
# Import dependencies
import pandas as pd


def create_tweets_df():
    cols = ["sentiment", "ids", "date", "flag", "user", "text"]
    encoding = "ISO-8859-1"

    return pd.read_csv('tweets.csv', encoding=encoding, names=cols)


def remove_redundant_cols(dataset):
    # Remove all columns apart from text and sentiment
    return dataset[['text', 'sentiment']]


def rename_values_in_sentiment_col(dataset):
    # Replacing the '4' value to '1' to symbolise positive tweets
    dataset["sentiment"] = dataset['sentiment'].replace(4, 1)
    return dataset


def plot_sentiment_distribution(dataset):
    dataset \
        .groupby('sentiment') \
        .count() \
        .plot(kind='bar', title="Distribution of Sentiment", legend=False)


def store_dataset_in_list(dataset):
    return list(dataset['text']), list(dataset['sentiment'])


def clean_and_import_tweet_df():
    tweets_df = create_tweets_df()
    tweets_df_with_removed_cols = remove_redundant_cols(tweets_df)
    return rename_values_in_sentiment_col(tweets_df_with_removed_cols)
