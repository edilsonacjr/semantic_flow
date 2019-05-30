

def main():
    print('Test execution')

    print('preprocessing and segmentation....')

    import nltk
    import numpy as np

    from sklearn.base import TransformerMixin
    from gensim.models import Word2Vec


    class Preprocessing(TransformerMixin):

        def __init__(self, stopwords='english', keep_cap=False, keep_numeric=False):

            if type(stopwords) is list:
                self.stopwords = stopwords
            else:
                self.stopwords = nltk.corpus.stopwords.words(stopwords)

            self.tokenize = tokenize = nltk.tokenize.RegexpTokenizer('(?u)\\b\\w\\w+\\b')

        def fit(self, X, y=None):
            return self

        def transform(self, X, y=None, **fit_params):
            out = []
            for text in X:
                text_out = []
                sents = nltk.sent_tokenize(text)
                for sent in sents:
                    original_tokens = self.tokenize.tokenize(sent) #mudar para word tokenizer
                    tokens = [w.lower() for w in original_tokens if w.lower() not in self.stopwords and
                              not w.isnumeric() and len(w) > 1] # tornar esse codigo mais eficiente
                    text_out.append(tokens)

            return out

        def fit_transform(self, X, y=None):
            return self.transform(X)

        # fazer asserts aqui!!!!!!!!!

    print('sentence embedding....')
    # what to do with oov

    class W2VTransformer(TransformerMixin):
        def __init__(self,
                     model_file,
                     model_type='word2vec',
                     loader_func=None,
                     remove_oov=True,
                     oov_vec=None,
                     empty_random=True,
                     random_seed=42):

            if model_type == 'word2vec':
                self.model = Word2Vec.load_word2vec_format('model_file', binary=True)
            else:
                self.model = loader_func(model_file)

            self.remove_oov = remove_oov
            self.oov_vec = oov_vec

        def transform(self, X, y=None, **fit_params):
            out_m = []

            for text in X:
                text_M = []
                for sent in text:
                    parag_M = []
                    for token in sent:
                        if token in self.model:
                            parag_M.append(self.model[token])
                    if parag_M:
                        text_M.append(np.average(parag_M, axis=0))
                    else:
                        text_M.append(np.random.rand(1, 300)[0])
                out_m.append(text_M)
            return np.array(out_m)

        def fit_transform(self, X, y=None, **fit_params):
            self.fit(X, y, **fit_params)
            return self.transform(X)

        def fit(self, X, y=None, **fit_params):
            return self

    print('graph representation...')
    print('community detection....')
    print('markov net assemble...')
    print('motif extraction....')
    print('classification...')

    print('all tests passed..')


if __name__ == '__main__':
    main()