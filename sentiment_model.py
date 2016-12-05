import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pandas as pd
from sklearn import model_selection
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def determine_sentiment(delta):
    return delta > 0


def remove_stop_words(text):
    global stop
    global porter
    words = [porter.stem(item.lower()) for item in text.split()
             if item not in stop]
    return ' '.join(words)

#load the data
data = pd.read_csv('./data/training_data.csv', encoding='latin1')
#data = pd.read_csv('./data/first10k.csv', encoding='latin1')

# sentiment feature generation
data['text'] = data['article_content'].apply(remove_punctuation)
stop = set(stopwords.words('english'))
porter = PorterStemmer()
data['text'] = data['text'].apply(remove_stop_words)

data['price_delta'] = data['close_31'] - data['open_31']
data['price_delta_percent'] = \
    ((data['close_31'] - data['open_31']) / data['open_31']) * 100


sentiment_data = data[(data['price_delta_percent'] >= 2) |
                      (data['price_delta_percent'] <= -2)]


sentiment_data['sentiment'] = sentiment_data['price_delta_percent'].apply(determine_sentiment)

count_vect = CountVectorizer()
# data['word_counts'] = count_vect.fit_transform(data['text'].values)
counts = count_vect.fit_transform(sentiment_data['text'].values)

# create the train / test split
train_X, test_X = model_selection.train_test_split(counts,
                                                   train_size=0.7,
                                                   random_state=0)

train_Y, test_Y = model_selection.train_test_split(sentiment_data['sentiment'],
                                                   train_size=0.7,
                                                   random_state=0)

model = MultinomialNB().fit(train_X, train_Y)



test_predictions = model.predict(test_X)

accuracy = accuracy_score(test_Y, test_predictions) * 100
print("Fully Trained Accuracy: {accuracy:.3f}".format(accuracy=accuracy))

