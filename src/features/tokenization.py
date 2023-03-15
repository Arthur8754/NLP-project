from nltk.tokenize import sent_tokenize, word_tokenize

class tokenization:
    """ 
    Cette classe permet de transformer une phrase en un ensemble de tokens.
    """

    def __init__(self) -> None:
        pass
    
    def tokenize_tweet(self, tweet):
        """ 
        Tokenize le tweet passé en paramètre, sous forme de listes de tokens.
        """
        tokenized_tweet = []

        # Découpage du tweet en phrases
        split_sentences = sent_tokenize(tweet)
        for sentence in split_sentences:

            # Découpage d'une phrase en tokens
            tokenized_sentence = word_tokenize(sentence)
            for token in tokenized_sentence:
                tokenized_tweet.append(token)

        return tokenized_tweet

