import string

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


def remove_punctuation(text):
    """Removes punctuation from the supplied string"""
    return text.translate(str.maketrans('', '', string.punctuation))


def remove_stop_words(text):
    """Removes stop words from the supplied text whilst stemming them"""
    stop = set(stopwords.words('english'))
    porter = PorterStemmer()

    words = [porter.stem(item.lower()) for item in text.split()
             if item not in stop]
    return ' '.join(words)
