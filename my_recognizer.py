import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    probabilities = []
    guesses = []

    for key in test_set.get_all_Xlengths():
        word_probabilities = {}

        X, length = test_set.get_all_Xlengths()[key]

        highest_logL = float('-inf')
        possible_word = ""

        word_probabilities = {}

        for word in models.keys():
            try:
                logL = models[word].score(X, length)
            except:
                logL = float('-inf')

            if logL > highest_logL:
                highest_logL = logL
                possible_word = word

            word_probabilities[word] = logL

        probabilities.append(word_probabilities)
        guesses.append(possible_word)

    return probabilities, guesses
