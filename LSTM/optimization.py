import sys, os, re, csv, codecs, numpy as np, pandas as pd, warnings

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, concatenate, SpatialDropout1D, GRU
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D, ActivityRegularization
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.callbacks import TensorBoard, Callback
from metrics import f1,recall, precision
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV



path = './'
EMBEDDING_FILE=f'{path}glove.6B.50d.txt'
TRAIN_DATA_FILE=f'{path}train.csv'
TEST_DATA_FILE=f'{path}test.csv'

embed_size = 50 # how big is each word vector
max_features = 100000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 150 # max number of words in a comment to use

train = pd.read_csv(TRAIN_DATA_FILE)
test = pd.read_csv(TEST_DATA_FILE)

tbCallBack = TensorBoard(log_dir='./tensorboard', histogram_freq=1, write_graph=True, write_images=True)

list_sentences_train = train["comment_text"].fillna("_na_").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("_na_").values

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.strip().split()) for o in open(EMBEDDING_FILE, encoding="utf8"))

all_embs = np.stack(embeddings_index.values())
emb_mean,emb_std = all_embs.mean(), all_embs.std()
emb_mean,emb_std

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

def create_model(LSTM_neurons=50, l2_rate = 0.001, dropout = 0.1):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(LSTM(LSTM_neurons, return_sequences=True, dropout=dropout, recurrent_dropout=0.15,
                           activity_regularizer=regularizers.l2(l2_rate)))(x)
    # x = Conv1D(64, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(x)
    x = Dropout(dropout)(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    x = Dense(6, activation="sigmoid", activity_regularizer=regularizers.l2(l2_rate))(conc)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', f1, recall, precision])
    return model


model = KerasClassifier(build_fn=create_model, epochs=2, batch_size=32, verbose=0)

LSTM_neurons = [20,30,40,50,60,70,80]
L2_rate = [10**i for i in range(-5,2)]
dropout = np.arange(0.0, 1.0, 0.05)

param_grid = dict(LSTM_neurons=LSTM_neurons, l2_rate=L2_rate, dropout=dropout )
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(X_t, y, batch_size=32, epochs=4, validation_split=0.25, callbacks=[tbCallBack]);

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


