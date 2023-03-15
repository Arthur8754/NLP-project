from src.features.tokenization import tokenization
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class tweetLevel:
    """ 
    Cette classe extrait les caractéristiques au niveau du tweet, en considérant chaque tweet indépendant des autres.
    """

    def __init__(self) -> None:
        pass

    def get_length_in_characters(self, tweet):
        """ 
        Donne la longueur du tweet, en caractères.
        """
        return len(tweet)
    
    def get_length_in_tokens(self, tweet):
        """ 
        Donne la longueur du tweet, en tokens.
        """
        tokenizer = tokenization()
        return len(tokenizer.tokenize_tweet(tweet))
    
    def get_positive_sentiment_score(self, tweet):
        """ 
        Donne le score de sentiment positif du tweet, c'est-à-dire la probabilité que le tweet soit positif.
        """
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(tweet)
        return scores["pos"]
