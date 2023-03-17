from src.features.tokenization import tokenization
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tag import pos_tag
import spacy

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
    
    def get_pos_tags(self, tweet):
        """ 
        Donne les POS tags de chaque token du tweet passé en paramètre.
        """
        # Tokenisation du tweet
        tokenizer = tokenization()
        tokenized_tweet = tokenizer.tokenize_tweet(tweet)
        return pos_tag(tokenized_tweet)
    
    def get_entity_types(self, tweet):
        """ 
        Donne les entités contenues dans le tweet, s'il y en a. 
        """
        entities = []
        nlp = spacy.load("en_core_web_sm")
        analyzed_tweet = nlp(tweet)
        for entity in analyzed_tweet.ents:
            entities.append((entity.text,entity.label_))
        return entities


