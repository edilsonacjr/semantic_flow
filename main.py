

import os
import sys
import argparse
import json
import subprocess


import nltk
import numpy as np
import pandas as pd

from collections import defaultdict

from sklearn.metrics.pairwise import euclidean_distances

from igraph import Graph, ADJ_UNDIRECTED
from igraph import arpack_options
from gensim.models import KeyedVectors

from utils import to_xnet
from utils import extract_motif
from utils import extract_weighted_motif

# bug on ARPACK
arpack_options.maxiter=300000


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


def generate_net_bert(texts_sents, bert_dir, book_names, sent_dir, net_dir, save_nets=True):
    nets = []
    for sents, book_name in zip(texts_sents, book_names):
        with open(os.path.join(sent_dir, book_name), 'w') as sent_file:
            for sent in sents:
                sent_file.write(' '.join(sent).replace('\n', ' ') + '\n')
        M = []

        # run BERT with the official code from the paper

        bert_cmd = ['python',
                    'bert/extract_features.py',
                    '--input_file=%s' % os.path.join(sent_dir, book_name),
                    '--output_file=/tmp/out_bert.jsonl',
                    '--vocab_file=%s/vocab.txt' % bert_dir,
                    '--bert_config_file=%s/bert_config.json' % bert_dir,
                    '--init_checkpoint=%s/bert_model.ckpt' % bert_dir,
                    '--layers=-1',
                    '--max_seq_length=128',
                    '--batch_size=8',
                    ]

        process = subprocess.Popen(bert_cmd, stdout=subprocess.PIPE)
        process.wait()

        # load BERT generated features
        with open('/tmp/out_bert.jsonl') as vectors_file:
            for line in vectors_file:
                feature_json = json.loads(line)
                M.append(feature_json['features'][0]['layers'][0]['values'])

        eucl = euclidean_distances(M)
        del M

        simi_m = 1. / (1. + eucl)

        k = 1
        while True:
            simi = simi_m.copy()
            to_remove = simi.shape[0] - (k + 1)

            for vec in simi:
                vec[vec.argsort()[:to_remove]] = 0

            g = Graph.Weighted_Adjacency(simi.tolist(), mode=ADJ_UNDIRECTED, loops=False)

            if g.is_connected():
                break
            k += 1

        del simi_m

        if save_nets:
            to_xnet(g, os.path.join(net_dir, book_name), names=False)
            g.write_pajek(os.path.join(net_dir, 'net_' + book_name + '.net'))

        nets.append(g)

    return nets


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
                    sent_file.write(' '.join(sent).replace('\n', ' ') + '\n')
                    M.append(np.average(parag_M, axis=0))

        eucl = euclidean_distances(M)
        del M

        simi_m = 1. / (1. + eucl)

        k = 1
        while True:
            simi = simi_m.copy()
            to_remove = simi.shape[0] - (k + 1)

            for vec in simi:
                vec[vec.argsort()[:to_remove]] = 0

            g = Graph.Weighted_Adjacency(simi.tolist(), mode=ADJ_UNDIRECTED, loops=False)

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
        if method == 'community_multilevel' or method == 'community_leading_eigenvector':
            comm = getattr(g, method)()
        else:
            comm = getattr(g, method)()
            comm = comm.as_clustering()
        y_pred = comm.membership

        if save_labels:
            np.savetxt(os.path.join(net_dir, book_name + '_labels.txt'), y_pred, fmt='%d')

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


def motif_extraction(networks, cuts, book_names, motif_dir, save_motifs=True):
    motifs = defaultdict(list)
    weighted_motif = defaultdict(list)

    cuts = np.append(['full'], cuts)
    for book_nets, book_name in zip(networks, book_names):
        for individual_net, cut in zip(book_nets, cuts):
            motif_freq, motif_perce, weighted_motifs = extract_weighted_motif(individual_net)
            motif_freq, motif_perce = extract_motif(individual_net)

            motifs[cut].append(motif_freq + [book_name])
            weighted_motif[cut].append(weighted_motifs + [book_name])

    out_motifs = []
    out_weighted = []
    if save_motifs:
        for cut in cuts:
            df_motif = pd.DataFrame(motifs[cut])
            df_motif.to_csv(os.path.join(motif_dir, 'extracted_' + str(cut) + '.csv'), index=False)
            out_motifs.append(df_motif)

            df_weighted = pd.DataFrame(weighted_motif[cut])
            df_weighted.to_csv(os.path.join(motif_dir, 'extracted_weighted_' + str(cut) + '.csv'),
                               index=False)
            out_weighted.append(df_weighted)

    return out_motifs, out_weighted


def main(args):
    # Downloading NLTK data
    nltk.download('punkt')

    print('load data')
    texts, labels = load_data(args.book_list_file, args.label_list_file, args.book_dir)

    if args.encoding_method == 'word2vec':
        print('load word embeddings')
        model = KeyedVectors.load_word2vec_format(args.word2vec_file, binary=True)
    elif args.encoding_method == 'bert':
        print('BERT selected')
    else:
        raise ValueError('Encoding method not implemented.')

    # Testing pipeline without loading a real word embeddings
    # from collections import defaultdict
    # model = defaultdict(lambda: np.random.rand(1, 300)[0])

    with open(args.book_list_file, 'r') as books_f:
        book_list = books_f.read().split()

    cuts = np.arange(args.range_cut_begin, args.range_cut_end, args.range_cut_step)
    print('prep data')
    texts_sents = prep_text(texts)

    print('generate_net')
    if args.encoding_method == 'word2vec':
        nets = generate_net(texts_sents, model, book_list, args.sent_dir, args.net_dir, args.save_nets)
    elif args.encoding_method == 'bert':
        nets = generate_net_bert(texts_sents, args.bert_dir, book_list, args.sent_dir, args.net_dir, args.save_nets)

    print('detect_community')
    comm_labels = detect_community(nets, args.comm_method, book_list, args.net_dir, args.save_labels)

    print('generate_markov')
    markov_nets = generate_markov(comm_labels, cuts, book_list, args.markov_dir, args.save_markov)

    print('motif_extraction')
    extracted_motifs = motif_extraction(markov_nets, cuts, book_list, args.motif_dir, args.save_motifs)

    print('end')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--book_list_file', type=str, help='File with the list of books. One in each line.',
                        default='book_list.txt')
    parser.add_argument('--label_list_file', type=str, help='File with the list of labels. One in each line.',
                        default='label_list.txt')
    parser.add_argument('--book_dir', type=str, help='Directory where the file of each book is stored.',
                        default='data/')
    parser.add_argument('--log_file', type=str, help='File to store log data.', default='log.txt')
    parser.add_argument('--encoding_method', type=str, choices=['word2vec', 'bert'],
                        help='Encoding method of the sentences.', default='word2vec')
    parser.add_argument('--word2vec_file', type=str, help='File to store log data.', default='embedding.bin')
    parser.add_argument('--bert_dir', type=str, help='BERT pretrained model.', default='bert/uncased_L-12_H-768_A-12')
    parser.add_argument('--sent_dir', type=str, help='Directory to save net format', default='sent/')
    parser.add_argument('--net_dir', type=str, help='Directory to save net format', default='net/')
    parser.add_argument('--save_nets', help='Saves all networks in xnet format.', action='store_true')
    parser.add_argument('--save_labels', help='Saves all detected communities in a txt file.', action='store_true')
    parser.add_argument('--comm_method', type=str, choices=['community_multilevel', 'community_leading_eigenvector',
                                                            'community_fastgreedy', 'community_walktrap'],
                        help='Community method to use.', default='community_multilevel')
    parser.add_argument('--markov_dir', type=str, help='Directory to save net format', default='markov/')
    parser.add_argument('--save_markov', help='Saves all markov nets.', action='store_true')
    parser.add_argument('--range_cut_begin', type=float, help='Markov range cut (begin)', default=0.01)
    parser.add_argument('--range_cut_end', type=float, help='Markov range cut (end)', default=0.205)
    parser.add_argument('--range_cut_step', type=float, help='Markov range cut (end)', default=0.005)
    parser.add_argument('--save_motifs', help='Saves all motifs.', action='store_true')
    parser.add_argument('--motif_dir', type=str, help='Directory to save motifs in csv format', default='motifs/')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
