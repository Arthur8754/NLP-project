from src.features.tokenization import tokenization
import os
import openai

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
    
    def get_sentiment_tweet(self, tweet):
        """ 
        Donne le score de sentiment du tweet. On exploite le playground de OpenAI (https://platform.openai.com/examples/default-tweet-classifier)
        """
        # Récupérer l'API KEY : 
        with open("../OpenAI-settings/API-key.txt") as file:
            openai.api_key = file.read().split("\n")[0]

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt = f"Give me only the probability the sentiment's tweet to be positive. Tweet:{tweet}",
            temperature=0,
            max_tokens=100,
            top_p=1.0,
            frequency_penalty = 0.5,
            presence_penalty=0.0
        )
        return response
