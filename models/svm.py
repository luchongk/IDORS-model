import numpy as np

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from pydmd import HODMD

class SVM:
    def __init__(self):
        self.clf = make_pipeline(StandardScaler(),
                                 LinearSVC(random_state=0, tol=1e-5, max_iter=50000))

    def fit(self, word_vectors, sent_vectors, bert_vectors, labels):
        final_vectors = np.zeros((len(sent_vectors), word_vectors[0].shape[1]*5 + sent_vectors.shape[1] + bert_vectors.shape[1]))
        for index, v in enumerate(word_vectors):
            list_of_modes = self._get_modes_from_word_vecs(v)
            list_of_modes.append(sent_vectors[index])
            list_of_modes.append(bert_vectors[index])
            final_vectors[index] = np.hstack(list_of_modes)

        self.clf.fit(final_vectors, labels)

    def evaluate(self, word_vectors, sent_vectors, bert_vectors, labels):
        final_vectors = np.zeros((len(sent_vectors), word_vectors[0].shape[1]*5 + sent_vectors.shape[1] + bert_vectors.shape[1]))
        for index, v in enumerate(word_vectors):
            list_of_modes = self._get_modes_from_word_vecs(v)
            list_of_modes.append(sent_vectors[index])
            list_of_modes.append(bert_vectors[index])
            final_vectors[index] = np.hstack(list_of_modes)
        
        predictions = self.clf.predict(final_vectors)
        tp, tn, fp, fn = 0, 0, 0, 0
        for i, p in enumerate(predictions):
            if labels[i] == 1:
                if p == 1:
                    tp += 1
                if p == 0:
                    fn += 1
            else:
                if p == 1:
                    fp += 1
                if p == 0:
                    tn += 1
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)       
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        fscore = 2 * (precision * recall) / (precision + recall)

        return accuracy, precision, recall, fscore, tp, tn, fp, fn

    def predict(self, word_vectors, sent_vectors, bert_vectors):
        final_vectors = np.zeros((len(sent_vectors), word_vectors[0].shape[1]*5 + sent_vectors.shape[1] + bert_vectors.shape[1]))
        for index, v in enumerate(word_vectors):
            list_of_modes = self._get_modes_from_word_vecs(v)
            list_of_modes.append(sent_vectors[index])
            list_of_modes.append(bert_vectors[index])
            final_vectors[index] = np.hstack(list_of_modes)
        
        return self.clf.predict(final_vectors)
    
    def save_weights(self, filename):
        dump(self.clf, filename)

    def load_weights(self, filename):
        return load(filename)

    def _get_modes_from_word_vecs(self, v, concat_avg = True):
        list_of_modes = []
        time_lags = [1,2]
        for d in time_lags:
            dmd = HODMD(svd_rank=2, opt=True, exact=True, d=d)
            dmd.fit(v.T)
            fmode = dmd.modes.T
            list_of_modes.append(np.hstack(np.absolute(fmode)))

        if concat_avg:
            mean_vec = self._p_mean_vector([1.0], v)
            list_of_modes.append(mean_vec)

        return list_of_modes
        
    
    def _p_mean_vector(self, powers, vectors):
        embeddings = []
        for p in powers:
            embeddings.append(np.power(np.mean(np.power(np.array(vectors, dtype=complex),p),axis=0),1 / p).real)
        return np.hstack(embeddings)

