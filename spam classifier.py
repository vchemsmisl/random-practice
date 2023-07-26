from preprocessor import Corpus, EMailText, create_count_vectors
from pathlib import Path
import json
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report


def main() -> None:
    vectors_path = Path(__file__).parent / 'vectors.json'
    if not vectors_path.exists():
        ham_path = Path(__file__).parent / 'corpora' / 'hard_ham'
        spam_path = Path(__file__).parent / 'corpora' / 'spam_2'
        ham_corpus = Corpus(ham_path, 'ham', 500)
        spam_corpus = Corpus(spam_path, 'spam', 500)
        for ham in ham_corpus.extract_links():
            text = EMailText(ham, ham_corpus)
            text.process(True, True)
        for spam in spam_corpus.extract_links():
            text = EMailText(spam, spam_corpus)
            text.process(True, True)
        vectorizer = create_count_vectors(ham_corpus, spam_corpus)
        ham_matrix = ham_corpus.create_feature_matrix(vectorizer)
        spam_matrix = spam_corpus.create_feature_matrix(vectorizer)
        vectors_dict = {'ham vectors': ham_matrix.tolist(), 'spam vectors': spam_matrix.tolist()}
        vectors_path.touch()
        with open(vectors_path, 'w', encoding='utf-8') as file:
            json.dump(vectors_dict, file)
    else:
        with open(vectors_path, 'r', encoding='utf-8') as file:
            vectors_dict = json.load(file)
            ham_matrix = numpy.array(vectors_dict['ham vectors'])
            spam_matrix = numpy.array(vectors_dict['spam vectors'])
    data = numpy.r_[ham_matrix, spam_matrix]
    X_train, X_test, y_train, y_test = train_test_split(data[:, :-1],
                                                        data[:, -1],
                                                        test_size=0.4,
                                                        random_state=42)
    logreg = LogisticRegression(max_iter=600)
    scores = cross_val_score(logreg, X_train, y_train, scoring='f1', cv=5)
    print(f'Scores: {scores}\n'
          f'Mean score: {scores.mean()}\n'
          f'Std of scores: {scores.std()}')
    
    logreg.fit(X_train, y_train)
    predict = logreg.predict(X_test)
    print(classification_report(y_test, predict))


if __name__ == '__main__':
    main()
