import numpy as np
from data import train_data, test_data

# Kelime dağarcığı oluştur
def build_vocab(datasets):
    vocab = set()
    for data in datasets:
        for sentence in data.keys():
            for w in sentence.lower().split():
                vocab.add(w)
    return {w: i for i, w in enumerate(sorted(vocab))}

# Cümleleri indeks listesine çevir
def vectorize(data, w2i):
    X, y = [], []
    for sent, lbl in data.items():
        idxs = [w2i[w] for w in sent.lower().split() if w in w2i]
        X.append(idxs)
        y.append(1 if lbl else 0)
    return X, y

# Kullanıma hazır veri
word_to_ix = build_vocab([train_data, test_data])
X_train, y_train = vectorize(train_data, word_to_ix)
X_test, y_test = vectorize(test_data, word_to_ix)
