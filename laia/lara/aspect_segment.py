import copy
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize
from scipy.sparse import csr_matrix


class AspectSegment(object):
    _tf_cut = 5

    def __init__(self, documents, aspect_keywords, count_vectorizer=None,
                 sent_tokenizer=None):
        self._chi_table = None
        self._aspect_count = None
        self._aspect_word_count = None
        self._documents = documents
        self._count_vectorizer = count_vectorizer or CountVectorizer(stop_words="english")
        #self._vocabulary = self._count_vectorizer
        self._aspect_keywords = copy.deepcopy(aspect_keywords)
        self._sent_tokenizer = sent_tokenizer or sent_tokenize
        self._n_aspects = len(self._aspect_keywords)
        self._n_vocab = len(self._count_vectorizer.vocabulary_)
        self._chi_table = csr_matrix((self._n_aspects, self._n_vocab))

    # private methods
    def _chi_square_value(self, A, B, C, D, N):
        denominator = (A + C) * (B + D) * (A + B) * (C + D)
        if (denominator > 0) and ((A + B) > self._tf_cut):
            return N * (A * D - B * C) * (A * D - B * C) / denominator
        else:
            return 0.0  # problematic case; word not assigned
