

import nltk
import os
import numpy as np

from gensim.models import Word2Vec

# File with the list of books to be used.
BOOKS = 'book_list_CAT.txt'
BOOKS_DIR = 'data/cat/'

# Word embedding file, other models such as Glove can be used but the load method must change.
WORD2VEC_FILE = ''

# Log file
LOG_FILE = 'log.txt'


def prep_test(text):
    pass



def main():
    tokenize = nltk.tokenize.RegexpTokenizer('(?u)\\b\\w\\w+\\b')

    stopwords = nltk.corpus.stopwords.words('english')
    embedding_model = Word2Vec.load_word2vec_format('../GoogleNews-vectors-negative300.bin', binary=True)

    with open(BOOKS, 'r') as f_books:
        book_list = f_books.read().split()


    for book in book_list:
        with open(BOOKS_DIR+book, 'r') as book_file:
            text = book_file.read()

            M = []

            sentences = nltk.sent_tokenize(text)
            for sent in sentences:
                original_tokens = tokenize.tokenize(sent)

                # Set string to lower case
                # Filter of stopwords
                # Filter of numbers
                # Filter of string of size 1
                # Words not in the embedding model are discarded
                tokens = [w.lower() for w in original_tokens if w.lower() not in stopwords and
                          not w.isnumeric() and len(w) > 1 and w.lower() in embedding_model]

                parag_M = []

                for token in tokens:
                    parag_M.append(embedding_model[token])

                if parag_M:
                    sent_file.write(sent.replace('\n', ' ') + '\n\n')
                    M.append(np.average(parag_M, axis=0))


            # Log
            if os.path.exists(LOG_FILE):
                append_write = 'a'  # append if already exists
            else:
                append_write = 'w'

            with open(LOG_FILE, append_write) as log_file:
                log_file.write('BOOK: {}\n'.format(book))
                log_file.write('Num. of sentences: {}\n'.format(len(sentences)))
