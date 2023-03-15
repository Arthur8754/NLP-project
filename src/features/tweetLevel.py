from features.tokenization import tokenization

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
