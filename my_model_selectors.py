import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        lowest_bic = float('inf')
        best_model = None
        number_of_features = len(self.X[0])
        number_of_datapoints = len(self.X)

        try:
            for n_components in range(self.min_n_components, self.max_n_components + 1):

                model = self.base_model(n_components)
                logL = model.score(self.X, self.lengths)
                free_parameters = n_components ** 2 + 2 * number_of_features * n_components - 1
                bic = -2 * logL + free_parameters * np.log(number_of_datapoints)

                if bic < lowest_bic:
                    lowest_bic = bic
                    best_model = model

            return best_model

        except:
            if best_model:
                return best_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        highest_dic = float('-inf')
        best_model = None

        m = len(self.words)

        try:
            for n_components in range(self.min_n_components, self.max_n_components + 1):

                model = self.base_model(n_components)
                logL = model.score(self.X, self.lengths)

                sum_other_words_logL = 0.

                for other_word in self.hwords.keys():
                    if other_word != self.this_word:
                        other_word_x, other_word_length = self.hwords[other_word]
                        other_word_logL = model.score(other_word_x, other_word_length)
                        sum_other_words_logL += other_word_logL

                dic = logL - 1/(m - 1) * sum_other_words_logL

                if dic > highest_dic:
                    highest_dic = dic
                    best_model = model

            return best_model

        except:
            if best_model:
                return best_model



class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        split_method = KFold(n_splits=2,shuffle=True)
        highest_logL = float('-inf')
        best_model = None

        try:
            for n_components in range(self.min_n_components, self.max_n_components + 1):

                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):

                    x_train, lengths_train = combine_sequences(cv_train_idx, self.sequences)

                    model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000,
                                        random_state=self.random_state, verbose=False).fit(x_train, lengths_train)

                    x_test, lengths_test = combine_sequences(cv_test_idx, self.sequences)
                    logL = model.score(x_test, lengths_test)

                    if logL > highest_logL:
                        highest_logL = logL
                        best_model = model

            return best_model

        except:
            if best_model:
                return best_model