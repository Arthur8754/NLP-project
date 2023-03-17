from gensim.models import Word2Vec
from src.features.tokenization import tokenization

"""
https://radimrehurek.com/gensim/models/word2vec.html
"""

class word2vec:
    """ 
    Cette classe permet de construire les fonctions générant des embeddings word2vec de mots.
    """
    def __init__(self, tweets) -> None:
        """ 
        L'entraînement se fait lors de l'instanciation du modèle Word2Vec !
        """
        tokenized_tweets = []
        tokenizer = tokenization()
        for tweet in tweets:
            tokenized_tweets.append(tokenizer.tokenize_tweet(tweet))
        
        self.model = Word2Vec(sentences=tokenized_tweets, min_count=1)

    def predict(self, tweet):
        """ 
        Prédit la représentation word2vec de chaque mot du tweet.
        """
        # Tokenisation
        tokenizer = tokenization()
        tokenized_tweet = tokenizer.tokenize_tweet(tweet)

        # Word2vec representation
        word2vec_tweet = []
        for word in tokenized_tweet:
            word2vec_tweet.append(self.model.wv[word])
        return word2vec_tweet
