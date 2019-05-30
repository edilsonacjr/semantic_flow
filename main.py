

import os
import sys
import argparse


import nltk
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances

from igraph import Graph, ADJ_UNDIRECTED
from gensim.models import Word2Vec

# File with the list of books to be used.
BOOKS = 'book_list_CAT.txt'
BOOKS_DIR = 'data/cat/'

# Word embedding file, other models such as Glove can be used but the load method must change.
WORD2VEC_FILE = ''

# Log file
LOG_FILE = 'log.txt'



def main_old(args):
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


def load_data(book_list_file, label_list_file, book_dir):
    texts = []

    with open(book_list_file, 'r') as books_f:
        book_list = books_f.read().split()

    with open(label_list_file, 'r') as labels_f:
        labels = labels_f.read().split()

    for book in book_list:
        with open(os.path.join(book_dir, book), 'r') as book_file:
            text = book_file.read()
            texts.append(text)

    return texts, labels

def prep_text(text):
    tokenize = nltk.tokenize.RegexpTokenizer('(?u)\\b\\w\\w+\\b')
    stopwords = nltk.corpus.stopwords.words('english')

    sents = nltk.sent_tokenize(text)
    final_sents = []
    for sent in sents:

        original_tokens = tokenize.tokenize(sent)

        tokens = [w.lower() for w in original_tokens if w.lower() not in stopwords and
                  not w.isnumeric() and len(w) > 1]

        final_sents.append(tokens)

    return final_sents



def generate_net(sents, model, book_name, sent_dir):
    with open(os.path.join(sent_dir, book_name), 'w') as sent_file:
        M = []
        for sent in sents:
            parag_M = []

            for token in sent:
                parag_M.append(model[token])

            if parag_M:
                sent_file.write(' '.join(sent).replace('\n', ' ') + '\n\n')
                M.append(np.average(parag_M, axis=0))

    eucl = euclidean_distances(M)

    del M

    k = 1
    while True:
        simi_m = 1 / (1 + eucl)
        to_remove = simi_m.shape[0] - (k + 1)

        for vec in simi_m:
            vec[vec.argsort()[:to_remove]] = 0

        g = Graph.Weighted_Adjacency(simi_m.tolist(), mode=ADJ_UNDIRECTED, loops=False)

        if g.is_connected():
            break
        k += 1
    del simi_m




def main(args):

    texts, labels = load_data(args.book_list, args.label_list, args.book_dir)

    for text in texts:


    #texts, labels = load_data('book_list_CAT.txt', 'label_list_CAT.txt', 'data/livrosCategorias')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--book_list_file', type=str, help='File with the list of books. One in each line.',
                        default='book_list.txt')
    parser.add_argument('--label_list_file', type=str, help='File with the list of labels. One in each line.',
                        default='label_list.txt')
    parser.add_argument('--book_dir', type=str, help='Directory where the file of each book is stored.',
                        default='/data')
    parser.add_argument('--log_file', type=str, help='File to store log data.', default='log.txt')
    parser.add_argument('--word2vec_file', type=str, help='File to store log data.', default='embedding.bin')
    parser.add_argument('--net_dir', type=str, help='Directory to save net format', default='/nets')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv)
    main(args)
