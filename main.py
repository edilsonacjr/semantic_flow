

import os
import sys
import argparse


import nltk
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances

from igraph import Graph, ADJ_UNDIRECTED
from gensim.models import Word2Vec

from utils import to_xnet

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


def prep_text(texts):
    tokenize = nltk.tokenize.RegexpTokenizer('(?u)\\b\\w\\w+\\b')
    stopwords = nltk.corpus.stopwords.words('english')
    final_texts_sents = []

    for text in texts:
        sents = nltk.sent_tokenize(text)
        final_sents = []
        for sent in sents:

            original_tokens = tokenize.tokenize(sent)

            tokens = [w.lower() for w in original_tokens if w.lower() not in stopwords and
                      not w.isnumeric() and len(w) > 1]

            final_sents.append(tokens)
        final_texts_sents.append(final_sents)

    return final_texts_sents


def generate_net(texts_sents, model, book_names, sent_dir, net_dir, save_nets=True):
    nets = []
    for sents, book_name in zip(texts_sents, book_names):
        with open(os.path.join(sent_dir, book_name), 'w') as sent_file:
            M = []
            for sent in sents:
                parag_M = []

                for token in sent:
                    if token in model:
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

        if save_nets:
            to_xnet(g, os.path.join(net_dir, book_name), names=False)

        nets.append(g)

    return nets


def detect_community(nets, method, book_names, net_dir, save_labels=True):
    comm_labels = []
    for g, book_name in zip(nets, book_names):
        if method == 'community_multilevel':
            comm = getattr(g, method)()
        else:
            comm = getattr(g, method)()
            comm = comm.as_clustering()
        y_pred = comm.membership
        if save_labels:
            np.savetxt(os.path.join(net_dir, book_name, '_labels.txt'), y_pred, fmt='%d')

        comm_labels.append(y_pred)

    return comm_labels


def generate_markov(comm_labels, cuts, book_names, markov_dir, save_markov=True):

    all_markov_nets = []

    for comm, book_name in zip(comm_labels, book_names):
        markov_nets = []
        num_comm = len(set(comm))
        markov_m = np.zeros((num_comm, num_comm))

        for c_ind in range(0, len(comm) - 1):
            markov_m[comm[c_ind]][comm[c_ind + 1]] += 1

        for row in markov_m:
            row /= max(np.sum(row), 1)

        g = Graph.Weighted_Adjacency(markov_m.tolist())
        markov_nets.append(g)
        if save_markov:
            g.write_pajek(os.path.join(markov_dir, book_name + '.net'))

        for threshold in cuts:
            markov_c = markov_m.copy()

            thr_matrix = markov_c <= threshold
            markov_c[thr_matrix] = 0

            g = Graph.Weighted_Adjacency(markov_c.tolist())
            markov_nets.append(g)
            if markov_c.any():
                g.write_pajek(os.path.join(markov_dir, book_name + '_' + str(threshold) + '.net'))

        all_markov_nets.append(markov_nets)

        return all_markov_nets


def motif_extraction():
    motifs = []

    return motifs

def main(args):

    texts, labels = load_data(args.book_list_file, args.label_list_file, args.book_dir)

    model = None

    with open(args.book_list_file, 'r') as books_f:
        book_list = books_f.read().split()

    cuts = np.arange(args.range_cut_begin, args.range_cut_end, args.range_cut_step)

    texts_sents = prep_text(texts)

    nets = generate_net(texts_sents, model, book_list, args.sent_dir, args.net_dir, args.save_nets)

    comm_labels = detect_community(nets, args.comm_method, book_list, args.net_dir, args.save_labels)

    markov_nets = generate_markov(comm_labels, cuts, book_list, args.markov_dir, args.save_markov)

    #texts, labels = load_data('book_list_CAT.txt', 'label_list_CAT.txt', 'data/livrosCategorias')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--book_list_file', type=str, help='File with the list of books. One in each line.',
                        default='book_list.txt')
    parser.add_argument('--label_list_file', type=str, help='File with the list of labels. One in each line.',
                        default='label_list.txt')
    parser.add_argument('--book_dir', type=str, help='Directory where the file of each book is stored.',
                        default='data/')
    parser.add_argument('--log_file', type=str, help='File to store log data.', default='log.txt')
    parser.add_argument('--word2vec_file', type=str, help='File to store log data.', default='embedding.bin')
    parser.add_argument('--net_dir', type=str, help='Directory to save net format', default='nets/')
    parser.add_argument('--save_graphs', help='Saves all networks in xnet format.', action='store_true')
    parser.add_argument('--save_labels', help='Saves all detected communities in a txt file.', action='store_true')
    parser.add_argument('--markov_dir', type=str, help='Directory to save net format', default='markov/')
    parser.add_argument('--save_markov', help='Saves all markov nets.', action='store_true')
    parser.add_argument('--range_cut_begin', type=float, help='Markov range cut (begin)', default=0.01)
    parser.add_argument('--range_cut_end', type=float, help='Markov range cut (end)', default=0.205)
    parser.add_argument('--range_cut_step', type=float, help='Markov range cut (end)', default=0.005)

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv)
    main(args)
