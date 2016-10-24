import copy
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from scipy.sparse import csr_matrix


class Document(object):
    def __init__(self, doc, sent_tokenizer=None, lemmatizer=None):
        self._doc = doc
        self._sents = self._docs_to_sentences(sent_tokenizer, lemmatizer)
        self._aspect = {(i, j): None for i in range(self.n_docs()) for j in range(self.n_sentences(i))}

    def n_docs(self):
        return len(self._doc)

    def n_sentences(self, i):
        n_docs = self.n_docs()
        if i<n_docs:
            return len(self._doc[i])
        else:
            raise AttributeError("Given index i is larger than number of documents %d." % n_docs)

    def aspect(self, doc_i, sent_j):
        try:
            aspect = self._aspect[(doc_i, sent_j)]
            return aspect
        except KeyError:
            raise AttributeError("Given index for doc and sent not within bounds.")

    def sentence(self, doc_i, sent_j):
        return self._sents[doc_i][sent_j]

    def document(self, doc_i):
        return self._sents[doc_i]

    def set_aspect(self, doc_i, sent_j, aspect_id):
        try:
            self._aspect[(doc_i, sent_j)] = aspect_id
        except KeyError:
            raise AttributeError("Given index for doc and sent not within bounds.")

    def _docs_to_sentences(self, sent_tokenizer=None, lemmatizer=None):
        _sent_tokenize = sent_tokenizer or sent_tokenize
        _lemmatizer = lemmatizer or WordNetLemmatizer().lemmatize
        return [[[_lemmatizer(w) for w in sent] for sent in _sent_tokenize(d)] for d in self._doc]


class AspectSegment(object):
    _tf_cut = 5

    def __init__(self, documents, aspect_keywords, count_vectorizer=None,
                 sent_tokenizer=None, lemmatizer=None):
        self._chi_table = None
        self._aspect_count = None
        self._aspect_word_count = None
        self._documents = Document(documents, sent_tokenizer, lemmatizer)
        self._count_vectorizer = count_vectorizer or\
            self.count_vectorizer(lemmatizer=lemmatizer)
        self._aspect_keywords = [set([lemmatizer(w) for w in keywords]) for keywords in aspect_keywords]
        self._sent_tokenizer = sent_tokenizer or sent_tokenize
        self._n_aspects = len(self._aspect_keywords)
        self._n_vocab = len(self._count_vectorizer.vocabulary_)
        self._chi_table = csr_matrix((self._n_aspects, self._n_vocab))

    def annotate(self, doc_i):
        for sent_j, sent in enumerate(self._documents[doc_i]):
            max_count = 0
            aspect_id = -1
            for index, keyword_set in enumerate(self._aspect_keywords):
                count = self.aspect_count(sent, keyword_set)
                if (count>max_count):
                    max_count = count
                    aspect_id = index
                elif (count==max_count):
                    aspect_id = -1
            self._documents.set_aspect(doc_i, sent_j, aspect_id)

    def aspect_count(self, sent, keywords):
        count = 0
        for w in sent:
            count += 1 if w in keywords else 0
        return count

    # private methods
    def _chi_square_value(self, A, B, C, D, N):
        denominator = (A + C) * (B + D) * (A + B) * (C + D)
        if (denominator > 0) and ((A + B) > self._tf_cut):
            return N * (A * D - B * C) * (A * D - B * C) / denominator
        else:
            return 0.0  # problematic case; word not assigned


    def count_vectorizer(self, min_df=2, max_df=0.95, tokenizer=None,
                         lemmatizer=None, preprocessor=None, max_features=None,
                         stop_words="english"):
        lemmatizer = lemmatizer or WordNetLemmatizer().lemmatize
        tokenizer = tokenizer or word_tokenize

        def _tokenizer(doc):
            words = tokenizer(doc)
            return [lemmatizer(w) for w in words]

        return CountVectorizer(stop_words=stop_words, tokenizer=_tokenizer,
                               preprocessor=preprocessor, min_df=min_df,
                               max_df=max_df, max_features=max_features)

