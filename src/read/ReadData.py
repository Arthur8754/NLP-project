import pandas as pd

class ReadData:
    """ 
    Cette classe lit les ensembles d'entraînement et de test, et effectue le prétraitement nécessaire pour avoir un ensemble tweet | target.
    """
    def __init__(self, train_path, test_path) -> None:
        """ 
        train_path : chemin vers l'ensemble d'entraînement.
        test_path : chemin vers l'ensemble de test.
        """
        self.train_path = train_path
        self.test_path = test_path

    def read_train(self):
        """ 
        Retourne un dataframe correspondant à l'ensemble d'entraînement.
        """
        return pd.read_csv(self.train_path)[["text","target"]]
    
    def read_test(self):
        """ 
        Retourne un dataframe correspondant à l'ensemble de test.
        """
        return pd.read_csv(self.test_path)[["text"]]