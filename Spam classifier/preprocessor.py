import numpy
import re
import en_core_web_sm
import codecs
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer


class WrongTypeError(Exception):
    """
    Raised if type isn't equal to "ham" or "spam"
    """


class EmptyDirectoryError(Exception):
    """
    Raised if directory is empty
    """


class EmptyFileError(Exception):
    """
    Raised if file is empty
    """


class Corpus:
    """
    Stores the whole corpus
    """

    def __init__(self, corpus_link: Path, text_type: str, max_num_docs: int):
        self._corpus_link = corpus_link
        self._files_list = []
        self._validate_corpus()
        if text_type != 'ham' and text_type != 'spam':
            raise WrongTypeError("type isn't equal to 'ham' or 'spam'")
        self._label = 0 if text_type == 'ham' else 1
        self.docs = []
        self._max_num_docs = max_num_docs

    def _validate_corpus(self) -> None:
        """
        Carries out checks on files' and directory's existence,
        on whether files or a directory are empty
        """
        if not self._corpus_link.exists():
            raise FileNotFoundError('file does not exist')
        if not self._corpus_link.is_dir():
            raise NotADirectoryError('path does not lead to directory')
        self._files_list = list(self._corpus_link.glob('*'))
        if not self._files_list:
            raise EmptyDirectoryError('the directory is empty')
        for file in self._files_list:
            if not file.stat().st_size:
                raise EmptyFileError('the file is empty')

    def extract_links(self) -> list:
        """
        Returns the given number of e-mails
        :return: list of e-mails
        """
        return self._files_list[:self._max_num_docs + 1]

    def create_feature_matrix(self, vectorizer: CountVectorizer) -> numpy.ndarray:
        """
        Creates a words x documents matrix with a Bag of Words method
        :param vectorizer: a class from scikit-learn module
        to convert a corpus to a matrix of tokens
        :return: matrix with words' vector representations
        """
        count_matrix = vectorizer.transform(self.docs).toarray()
        return numpy.c_[count_matrix,
                        numpy.array([self._label] * count_matrix.shape[0])]


class EMailText:
    """
    Reads link to text and stores e-mails' content
    """

    def __init__(self, link: Path, corpus: Corpus) -> None:
        self._link = link
        self._corpus = corpus
        self._nlp = en_core_web_sm.load()
        self.email_rawtext = ''
        self._stopwords = self._nlp.Defaults.stop_words

    def _extract_full_text(self) -> None:
        """
        Extracts raw text from the link
        """
        with codecs.open(str(self._link), 'r', encoding='utf-8', errors='ignore') as file:
            self.email_rawtext = file.read()


    def _tokenize_and_lemmatize_text(self) -> str:
        """
        Removes punctuation, splits text to tokens,
        replaces numbers and urls with special tokens,
        carries out lemmatization
        :param text: raw text
        :param lemmatize: a boolean value, indicating
        whether to lemmatize text
        :return: lemmatized text in string format
        """
        text = self.email_rawtext.lower()
        text = re.sub(r'https?://.*', 'URL', text)
        text = re.sub(r'\d+', ' NUM ', text)
        text = re.sub(r'[^A-Za-z\s]+', '', text)
        doc = self._nlp(text)
        return ' '.join([token.lemma_ for token in doc])

    def _delete_stop_words(self, text: str) -> str:
        """
        Removes stop words
        :return: text without stop words
        """
        wordlist = text.split()
        return ' '.join([token for token in wordlist if token not in self._stopwords])

    def process(self, lemmatize: bool, delete_stopwords: bool) -> None:
        """
        Carries out full text preprocessing
        :param lemmatize: whether to lemmatize text
        :param delete_stopwords:
        """
        self._extract_full_text()
        if not lemmatize:
            self._corpus.docs.append(self.email_rawtext)
        lemmatized_text = self._tokenize_and_lemmatize_text()
        if delete_stopwords:
            lemmatized_text = self._delete_stop_words(lemmatized_text)
        self._corpus.docs.append(lemmatized_text)


def create_count_vectors(*corpora,
                         ngram_range: tuple = (1, 1),
                         max_df: int = 1.0,
                         min_df: int = 1) -> CountVectorizer:
    """
    Fits the vectorizer's dictionary to create words' vector representations
    :param corpora: spam and ham e-mails collections
    :return: fitted vectorizer
    """
    docs = [doc for corpus in corpora for doc in corpus.docs]
    vectorizer = CountVectorizer(ngram_range=ngram_range,
                                 max_df=max_df,
                                 min_df=min_df)
    vectorizer.fit(docs)
    return vectorizer
