from sklearn import model_selection
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

import pandas as pd

from modelling.utils import remove_punctuation, remove_stop_words


def determine_sentiment(delta):
    """Returns 1 for positive sentiment, 0 otherwise"""
    if delta > 0:
        return 1
    else:
        return 0



def train_and_save_model(data_filename, model_filename):
    pd.options.mode.chained_assignment = None
    # load the data
    # data = pd.read_csv('./data/training_data.csv', encoding='latin1')
    # data = pd.read_csv('./data/first10k.csv', encoding='latin1')
    data = pd.read_csv(data_filename, encoding='latin1')

    # drop any invalid rows, if the data is incomplete
    data.dropna(inplace=True)

    # if we only keep a few categories we want:
    data = data.query('cat_id not in [40, 41, 42, 43, 98, 168]')

    # sentiment feature generation
    data['text'] = data['article_content'].apply(remove_punctuation)
    data['text'] = data['text'].apply(remove_stop_words)

    # generate price delta and labels
    data['price_delta'] = data['close_31'] - data['open_31']
    data['price_delta_percent'] = \
        ((data['close_31'] - data['open_31']) / data['open_31']) * 100

    data['sentiment'] = \
        data['price_delta_percent'].apply(determine_sentiment)

    # create the train / test split
    train_X, test_X = \
        model_selection.train_test_split(data['article_content'],
                                         train_size=0.7,
                                         random_state=0)

    train_Y, test_Y = model_selection.train_test_split(data['sentiment'],
                                                       train_size=0.7,
                                                       random_state=0)

    pipeline = Pipeline([('count_vectorizer', CountVectorizer(ngram_range=(1,
                                                                           3))),
                         ('tfidf_transformer', TfidfTransformer()),
                         ('classifier', MultinomialNB())])

    pipeline.fit(train_X, train_Y)

    test_predictions = pipeline.predict(test_X)

    accuracy = accuracy_score(test_Y, test_predictions) * 100
    print("Fully Trained Accuracy: {accuracy:.3f}".format(accuracy=accuracy))

    print('Saving model to {file}'.format(file=model_filename))
    joblib.dump(pipeline, model_filename)

    pd.options.mode.chained_assignment = 'warn'


def load_model():
    pass


def predict():
    pass