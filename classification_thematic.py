
import os
import sys
import argparse

import pandas as pd
import numpy as np


from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import normalize


def main(args):

    """ First classification: distinguishing three different thematic classes:
    (i) children books; (ii) investigative; and (iii) philosophy books"""

    with open(args.label_list_file, 'r') as labels_f:
        labels = labels_f.read().split()

    cuts = np.arange(args.range_cut_begin, args.range_cut_end, args.range_cut_step)
    cuts = np.append(['full'], cuts)

    # Tests
    tests = [('C', 'I'), ('C', 'P'), ('I', 'P'), ('C', 'I', 'P')]

    results_unweighted = {
        'classifier': [],
        'accuracy': [],
        'std': [],
        'cut': [],
        'subtask': []
    }
    # cuts unweighted
    for cut in cuts:

        df_motif = pd.read_csv(os.path.join(args.motif_dir, 'extracted_%s.csv' % str(cut)))

        # Drop last column, the column with the name of the books
        df_motif = df_motif.drop('13', axis=1)

        # Add label column
        df_motif['labels'] = labels

        for test in tests:

            X = np.concatenate([df_motif[df_motif['labels'] == x].drop('labels', axis=1) for x in test])
            X = normalize(X, axis=1, norm='l1')
            y = np.concatenate([[x] * df_motif[df_motif['labels'] == x].shape[0] for x in test])

            clf = DecisionTreeClassifier()
            scores = cross_validate(clf, X, y, cv=10, scoring='accuracy', n_jobs=-1)

            results_unweighted['classifier'].append('CART')
            results_unweighted['accuracy'].append(np.mean(scores['test_score']))
            results_unweighted['std'].append(np.std(scores['test_score']))
            results_unweighted['cut'].append(cut)
            results_unweighted['subtask'].append('x'.join(test))

            clf = KNeighborsClassifier(1)
            scores = cross_validate(clf, X, y, cv=10, scoring='accuracy', n_jobs=-1)

            results_unweighted['classifier'].append('kNN')
            results_unweighted['accuracy'].append(np.mean(scores['test_score']))
            results_unweighted['std'].append(np.std(scores['test_score']))
            results_unweighted['cut'].append(cut)
            results_unweighted['subtask'].append('x'.join(test))

            clf = SVC(kernel='linear')
            scores = cross_validate(clf, X, y, cv=10, scoring='accuracy', n_jobs=-1)

            results_unweighted['classifier'].append('SVM')
            results_unweighted['accuracy'].append(np.mean(scores['test_score']))
            results_unweighted['std'].append(np.std(scores['test_score']))
            results_unweighted['cut'].append(cut)
            results_unweighted['subtask'].append('x'.join(test))

            clf = GaussianNB()
            scores = cross_validate(clf, X, y, cv=10, scoring='accuracy', n_jobs=-1)

            results_unweighted['classifier'].append('NB')
            results_unweighted['accuracy'].append(np.mean(scores['test_score']))
            results_unweighted['std'].append(np.std(scores['test_score']))
            results_unweighted['cut'].append(cut)
            results_unweighted['subtask'].append('x'.join(test))
    
    df_full = pd.DataFrame(results_unweighted)
    for test in tests:
        df_full[df_full['subtask'] == 'x'.join(test)].to_csv(
            os.path.join(args.results_dir, 'unweighted_' + 'x'.join(test) + '.csv'),
            index=False
        )

    results_weighted = {
        'classifier': [],
        'accuracy': [],
        'std': [],
        'cut': [],
        'subtask': []
    }

    # cuts weighted
    for cut in cuts:

        df_motif = pd.read_csv(os.path.join(args.motif_dir, 'extracted_weighted_%s.csv' % str(cut)))

        # Drop last column, the column with the name of the books
        df_motif = df_motif.drop('13', axis=1)

        # Add label column
        df_motif['labels'] = labels

        for test in tests:

            X = np.concatenate([df_motif[df_motif['labels'] == x].drop('labels', axis=1) for x in test])
            y = np.concatenate([[x] * df_motif[df_motif['labels'] == x].shape[0] for x in test])

            clf = DecisionTreeClassifier()
            scores = cross_validate(clf, X, y, cv=10, scoring='accuracy', n_jobs=-1)

            results_weighted['classifier'].append('CART')
            results_weighted['accuracy'].append(np.mean(scores['test_score']))
            results_weighted['std'].append(np.std(scores['test_score']))
            results_weighted['cut'].append(cut)
            results_weighted['subtask'].append('x'.join(test))

            clf = KNeighborsClassifier(1)
            scores = cross_validate(clf, X, y, cv=10, scoring='accuracy', n_jobs=-1)

            results_weighted['classifier'].append('kNN')
            results_weighted['accuracy'].append(np.mean(scores['test_score']))
            results_weighted['std'].append(np.std(scores['test_score']))
            results_weighted['cut'].append(cut)
            results_weighted['subtask'].append('x'.join(test))

            clf = SVC(kernel='linear')
            scores = cross_validate(clf, X, y, cv=10, scoring='accuracy', n_jobs=-1)

            results_weighted['classifier'].append('SVM')
            results_weighted['accuracy'].append(np.mean(scores['test_score']))
            results_weighted['std'].append(np.std(scores['test_score']))
            results_weighted['cut'].append(cut)
            results_weighted['subtask'].append('x'.join(test))

            clf = GaussianNB()
            scores = cross_validate(clf, X, y, cv=10, scoring='accuracy', n_jobs=-1)

            results_weighted['classifier'].append('NB')
            results_weighted['accuracy'].append(np.mean(scores['test_score']))
            results_weighted['std'].append(np.std(scores['test_score']))
            results_weighted['cut'].append(cut)
            results_weighted['subtask'].append('x'.join(test))

    df_full = pd.DataFrame(results_weighted)
    for test in tests:
        df_full[df_full['subtask'] == 'x'.join(test)].to_csv(
            os.path.join(args.results_dir, 'weighted_' + 'x'.join(test) + '.csv'),
            index=False
        )


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--label_list_file', type=str, help='File with the list of labels. One in each line.',
                        default='label_list.txt')

    parser.add_argument('--results_dir', type=str, help='Directory to save final results', default='cls_results/')
    parser.add_argument('--range_cut_begin', type=float, help='Markov range cut (begin)', default=0.01)
    parser.add_argument('--range_cut_end', type=float, help='Markov range cut (end)', default=0.205)
    parser.add_argument('--range_cut_step', type=float, help='Markov range cut (end)', default=0.005)
    parser.add_argument('--motif_dir', type=str, help='Directory to save motifs in csv format', default='motifs/')

    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    main(args)
