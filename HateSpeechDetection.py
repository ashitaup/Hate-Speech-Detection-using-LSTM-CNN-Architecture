# Commented out IPython magic to ensure Python compatibility.
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, Dense, Dropout, LSTM, Bidirectional, GRU, Embedding, Masking, MaxPooling1D, Flatten
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import gensim

import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import re
from nltk.tokenize import word_tokenize
import nltk, string
nltk.download('punkt')
nltk.download('stopwords')

# %tensorflow_version 1.x
# %matplotlib inline

!ls

GLOVE_PATH = 'drive/My Drive/Colab Notebooks/data/glove/glove.840B.300d.txt'
word2vec = 'drive/My Drive/Colab Notebooks/data/word2vec/GoogleNews-vectors-negative300.bin'
DATA_PATH = 'drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/data'
MODEL_PATH = 'drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/models/best.hdf5'

from google.colab import drive
drive.mount('/content/drive')

train = pd.read_csv(f'{DATA_PATH}/train.csv')
test = pd.read_csv(f'{DATA_PATH}/test.csv')
# extra = pd.read_csv('drive/My Drive/Colab Notebooks/data/sexual_abuse_yt.csv')

extra.head()
len(train)

x_train = train['comment_text'].str.lower()
x_test = test['comment_text'].str.lower()
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

# extra = extra['Comment'].str.lower()

# GLOBALS
max_features = 300000
maxlen = 150
embed_size = 300

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
# words = [w for w in words if not w in stop_words]

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# import re
# 
# # Preparing input for custom embeddings
# sequences = []
# for line in x_train:
#     # Lowercase
#     line = line.lower()
#     # Remove whitespaces
#     line = line.strip()
#     # Remove numbers
#     line = re.sub(r'\d+', '', line)
# 
#     # Removing emojis
#     line = line.encode('ascii', 'ignore').decode('ascii')
#     # Split into words
#     from nltk.tokenize import word_tokenize
#     tokens = word_tokenize(line)
# 
#     # Remove punctuation from each word
#     import string
#     table = str.maketrans('', '', string.punctuation)
#     stripped = [w.translate(table) for w in tokens]
# 
#     # Remove remaining tokens that are not alphabetic
#     words = [word for word in stripped if word.isalpha()]
# 
#     # Filter out stop words
#     from nltk.corpus import stopwords
#     stop_words = set(stopwords.words('english'))
#     words = [w for w in words if not w in stop_words]
# 
#     sequences.append(words)
# 
# # for line in extra:
# #     # Lowercase
# #     line = line.lower()
# #     # Remove whitespaces
# #     line = line.strip()
# #     # Remove numbers
# #     line = re.sub(r'\d+', '', line)
# 
# #     # Removing emojis
# #     line = line.encode('ascii', 'ignore').decode('ascii')
# 
# #     # Split into words
# #     from nltk.tokenize import word_tokenize
# #     tokens = word_tokenize(line)
# 
# #     # Remove punctuation from each word
# #     import string
# #     table = str.maketrans('', '', string.punctuation)
# #     stripped = [w.translate(table) for w in tokens]
# 
# #     # Remove remaining tokens that are not alphabetic
# #     words = [word for word in stripped if word.isalpha()]
# 
# 
# 
#     # Filter out stop words
#     from nltk.corpus import stopwords
#     stop_words = set(stopwords.words('english'))
#     words = [w for w in words if not w in stop_words]
# 
#     sequences.append(words)

# %%time

from gensim.models import Word2Vec
custom = Word2Vec(sequences,
                 min_count=3,   # Ignore words that appear less than this
                 size=300,      # Dimensionality of word embeddings
                 workers=2,     # Number of processors (parallelisation)
                 window=5,      # Context window for words during training
                 iter=30)       # Number of epochs training over corpus

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# # Tokenizing and Padding
# import pickle
# 
# # tokenizer = text.Tokenizer(num_words=max_features, lower=True)
# # tokenizer.fit_on_texts(list(x_train) + list(x_test) + list(extra))
# 
# # # Storing tokenizer for future ref
# # with open('drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/models/tokenizer.pickle', 'wb') as f:
# #     pickle.dump(tokenizer, f)
# 
# with open('drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/models/tokenizer.pickle', 'rb') as f:
#     tokenizer = pickle.load(f)
# 
# x_train = tokenizer.texts_to_sequences(x_train)
# # extra = tokenizer.texts_to_sequences(extra)
# x_test = tokenizer.texts_to_sequences(x_test)
# 
# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# # extra = sequence.pad_sequences(extra, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# Embedding Index from Glove/word2vec vectors
embedding_index = gensim.models.KeyedVectors.load_word2vec_format(word2vec, binary=True)

# Building Embedding matrix
word_index = tokenizer.word_index
embedding_matrix = np.zeros((max_features, embed_size))

cnt = 0
for word, i in word_index.items():
    if i < max_features:
        if word in embedding_index:
            embedding_vector = embedding_index[word]
            embedding_matrix[i] = embedding_vector
        elif word in custom:
            cnt += 1
            embedding_vector = custom[word]
            embedding_matrix[i] = embedding_vector

# Model architecture
model = Sequential()

model.add(Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False, input_length=maxlen))
model.add(SpatialDropout1D(0.2))
model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.1, recurrent_dropout=0.1)))

model.add(Conv1D(64, 3))
model.add(GlobalAveragePooling1D())

model.add(Dense(6, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

batch_size = 128
epochs = 3
x_tra, x_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.6, random_state=42)

from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback

class roc_auc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict_proba(self.x, verbose=0)
        roc = roc_auc_score(self.y, y_pred)
        logs['roc_auc'] = roc_auc_score(self.y, y_pred)
        logs['norm_gini'] = ( roc_auc_score(self.y, y_pred) * 2 ) - 1

        y_pred_val = self.model.predict_proba(self.x_val, verbose=0)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        logs['roc_auc_val'] = roc_auc_score(self.y_val, y_pred_val)
        logs['norm_gini_val'] = ( roc_auc_score(self.y_val, y_pred_val) * 2 ) - 1

        print('\rroc_auc: %s - roc_auc_val: %s - norm_gini: %s - norm_gini_val: %s' % (str(round(roc,5)),str(round(roc_val,5)),str(round((roc*2-1),5)),str(round((roc_val*2-1),5))), end=10*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

# Callbacks
checkpoint = ModelCheckpoint(MODEL_PATH, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
callbacks_list = [roc_auc_callback(training_data=(x_tra, y_tra),validation_data=(x_val, y_val)), checkpoint, early]

x_tra.shape, x_val.shape

# %%time
# Model training...
batch_size = 128
epochs = 2
history = model.fit(
    x_tra,
    y_tra,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    callbacks=callbacks_list,
    verbose=1
)
model.load_weights(MODEL_PATH)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D #<-- Note the capitalization!

X = custom[custom.wv.vocab]
pca = PCA(n_components=3)
result = pca.fit_transform(X)

# Create a scatter plot of the projection
fig = plt.figure()
ax = Axes3D(fig) #<-- Note the difference from your original code...
ax.scatter3D(result[:100, 0], result[:, 1], result[:, 2], cmap='Greens')
words = list(custom.wv.vocab)
for i, word in enumerate(words):
	plt.annotate(word, xyz=(result[i, 0], result[i, 1]))
plt.show()

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# y_test = model.predict(x_test, batch_size=batch_size, verbose=1)

submission = pd.read_csv(f'{DATA_PATH}/sample_submission.csv')
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_test
# submission.to_csv(f'{DATA_PATH}/gru_cnn_custom_submission.csv', index=False)

submission.head()
# len(y_test), len(submission)

model = load_model('drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/models/v1.h5')

def preprocess(text, stemming=False):

    # Lowercase
    text = text.lower()
    # Remove whitespaces
    text = text.strip()
    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Removing emojis
    text = text.encode('ascii', 'ignore').decode('ascii')

    # Remove stopwords
    text = word_tokenize(text)
    # text = [i for i in text if not i in stop_words]

    # Remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    text = [t.translate(table) for t in text]

    # Stemming
    if stemming:
        text = [stemmer.stem(token) for token in text]

    # Removing empty strings
    text = [t for t in text if len(t.strip()) > 0]

    return text

#KillAllNiggers
#IStandWithHateSpeech
with open('drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/models/tokenizer.pickle', 'rb') as f:
    tokenizer = pickle.load(f)

def predict(x):
    x = ' '.join(preprocess(x))
    print(x)
    x = np.array([x])
    x = tokenizer.texts_to_sequences(x)
    x = sequence.pad_sequences(x, maxlen=maxlen)
    # print(x.shape)
    pred = model.predict(x)
    pred = pd.DataFrame(pred, columns=["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
    return pred

msg1 = "It’s not only girls that suffer from cyberbullying. While most horror stories involve girls, there are many young boys who have to deal with the torment and horrible words. Kenneth Weishuhn was one of those boys, who was bullied because of his sexual orientation. Being gay is hard as a teen without the bullying, but it’s worse when your classmates create an anti-gay Facebook group and make death threats over the phone. This wasn’t just from enemies or people he barely knew. After he “came out,”"
pred = predict(msg1)

pred

msg2 = "why is he saying This is a fucking broomstick being shoved down his urthrea"

pred2 = predict(msg2)
pred2

msg3 = "Hey, man this isnt a formal debate about blacks. We have to work on our our actual topic."
pred3=  predict(msg3)
pred3

model.save('drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/models/v1.h5')
model.save('drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/models/v2.h5', save_format='tf')

"""NEW STUFF STArts"""

#hie idk
import numpy as np
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from sklearn.metrics import classification_report
class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

DATA_PATH = 'drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/data'
train = pd.read_csv(f'{DATA_PATH}/train.csv').fillna(' ')
test = pd.read_csv(f'{DATA_PATH}/test.csv').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])

word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000 )
word_vectorizer.fit(all_text)

x_tra, x_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.6, random_state=42)

train_word_features = word_vectorizer.transform(x_tra)
test_word_features = word_vectorizer.transform(x_val)

train_char_features = char_vectorizer.transform(x_tra)
test_char_features = char_vectorizer.transform(x_val)

char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)
char_vectorizer.fit(all_text)

with open('drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/models/char_vec.pickle', 'wb') as f:
    pickle.dump(char_vectorizer, f)
with open('drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/models/word_vec.pickle', 'wb') as f:
    pickle.dump(word_vectorizer, f)

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])

scores = []
submission = pd.DataFrame.from_dict({'id': test['id']})

classifiers = {}
for i, class_name in enumerate(class_names):
    train_target = y_tra[:, i]
    classifier = LogisticRegression(C=1, solver='sag')

    cv_score = np.mean(cross_val_score(classifier, train_features, train_target, cv=2, scoring='roc_auc'))
    scores.append(cv_score)
    print('CV score for class {} is {}'.format(class_name, cv_score))

    classifier.fit(train_features, train_target)
    classifiers[class_name] = classifier
    # submission[class_name] = classifier.predict_proba(test_features)[:, 1]

print('Total CV score is {}'.format(np.mean(scores)))

with open('drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/models/lreg.pickle', 'wb') as f:
  pickle.dump(classifiers, f)

with open('drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/models/lreg.pickle', 'rb') as f:
  lregs = pickle.load(f)

with open('drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/models/char_vec.pickle', 'rb') as fp:
    char_vectorizer = pickle.load(fp)
with open('drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/models/word_vec.pickle', 'rb') as f:
    word_vectorizer = pickle.load(f)

sentence = x_val
char_sent = char_vectorizer.transform(sentence)
word_sent = word_vectorizer.transform(sentence)
my_testing = hstack([char_sent, word_sent])
my_testing

def testing_predictions(classifiers):
    sentence = x_val
    char_sent = char_vectorizer.transform(sentence)
    word_sent = word_vectorizer.transform(sentence)
    my_testing = hstack([char_sent, word_sent])
    pred = []
    for class_name in class_names:
        t = classifiers[class_name].predict(my_testing)
        pred.append(t)
    pred = np.array(pred)
    pred = pred.T

    return pred

rfpred = testing_predictions(classifiers_forest)
rfpred.shape

def print_every_metric(pred):
    print('Accuracy score:', accuracy_score(y_val, pred))
    print('Classification report:\n', classification_report(y_val, pred))
    print('ROC AUC score:', roc_auc_score(y_val, pred))

print_every_metric(pred)

# pred = []
# for class_name in class_names:
#   t = classifier_boost[class_name].predict(my_testing)
#   pred.append(t)
# len(pred)
# pred = np.array(pred)
# pred = pred.T
# pred.shape
# pred = classifier.predict(my_testing)
# classifier.predict(my_testing)
# train[5:10]
# pred

pred[:5]

y_tra.shape

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# classifiers_forest = {}
# for i, j in enumerate(class_names):
#     m = RandomForestClassifier(n_estimators=100, max_leaf_nodes=18, random_state=42)
#     print('fit', j)
#     m.fit(train_char_features, y_tra[:, i])
#     classifiers_forest[j] = m

# Transformed word - char vectors
with open('drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/models/random_forest.pickle', 'wb') as f:
    pickle.dump(classifiers_forest, f)

pred_forest = []
for class_name in class_names:
    t = classifiers_forest[class_name].predict(char_sent)
    pred_forest.append(t)
len(pred_forest)
pred_forest = np.array(pred_forest)
pred_forest = pred_forest.T
pred_forest.shape

print_every_metric(pred_forest)

classifiers['identity_hate'].predict_proba(my_testing)

for i in class_names:
  print(i, classifiers[i].predict_proba(my_testing))

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def auroc(y_true, y_pred):
    return tf.compat.v1.py_func(roc_auc_score, (y_true, y_pred), tf.double)
# evaluate the model

loss, accuracy, f1_score, precision, recall = model.evaluate(x_val, y_val, verbose=0)

loss, accuracy, f1_score, precision, recall

x_train = train['comment_text'].str.lower()
x_test = test['comment_text'].str.lower()
# x_tra, x_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.6, random_state=42)

#for logistic regression
print(classification_report(y_val,pred))

#for random forest
print(classification_report(y_val,pred_forest))

from sklearn.metrics import accuracy_score
# accuracy_score(y_val, pred_forest)

!pip install xgboost

# XGBoost stuff
# Imports
import xgboost as xgb # <3

# Commented out IPython magic to ensure Python compatibility.
# %%time
# # Training...
# 
# cv_scores = []
# classifier_boost = {}
# 
# for i, class_name in enumerate(class_names):
# 
#     xgb_params = {
#         'eta': 0.3,
#         'max_depth': 5,
#         'subsample': 0.8,
#         'colsample_bytree': 0.8,
#         'objective': 'binary:logistic',
#         'eval_metric': 'auc',
#         'seed': 23
#     }
# 
#     d_train = xgb.DMatrix(train_features, y_tra[:, i])
#     d_valid = xgb.DMatrix(test_features, y_val[:, i])
# 
#     watchlist = [(d_valid, 'valid')]
#     xgb_classifier = XGBClassifier.fit(xgb_params, d_train, 1, watchlist, verbose_eval=True, early_stopping_rounds=10)
#     print("class Name: {}".format(class_name))
#     print(xgb_classifier.attributes()['best_msg'])
#     cv_scores.append(float(xgb_classifier.attributes()['best_score']))
#     classifier_boost[class_name] = xgb_classifier
# 
# 
# print('Total CV score is {}'.format(np.mean(cv_scores)))

with open('drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/models/xgboost.pickle', 'wb') as f:
  pickle.dump(classifier_boost, f)

# with open('drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/models/xgboost.pickle', 'rb') as f:
#     classifier_boost = pickle.load(f)

boost_pred = testing_predictions(classifier_boost)
boost_pred.shape

# Transformed word - char vectors
with open('drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/models/word_train.pickle', 'wb') as f:
  pickle.dump(train_word_features, f)

# Transformed word - char vectors
with open('drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/models/word_test.pickle', 'wb') as f:
  pickle.dump(test_word_features, f)

# -------------------------------------------------------

# Transformed word - char vectors
with open('drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/models/char_train.pickle', 'wb') as f:
  pickle.dump(train_char_features, f)

# Transformed word - char vectors
with open('drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/models/char_test.pickle', 'wb') as f:
  pickle.dump(test_char_features, f)

# Transformed word - char vectors
with open('drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/models/word_train.pickle', 'rb') as f:
  train_word_features = pickle.load(f)

# Transformed word - char vectors
with open('drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/models/word_test.pickle', 'rb') as f:
  test_word_features = pickle.load(f)

# -------------------------------------------------------

# Transformed word - char vectors
with open('drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/models/char_train.pickle', 'rb') as f:
  train_char_features = pickle.load(f)

# Transformed word - char vectors
with open('drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/models/char_test.pickle', 'rb') as f:
  test_char_features = pickle.load(f)

sentence = x_val
char_sent = char_vectorizer.transform(sentence)
word_sent = word_vectorizer.transform(sentence)
my_testing = hstack([char_sent, word_sent])

with open('drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/models/xgboost.pickle', 'rb') as f:
  classifier_boost = pickle.load(f)

pred = []
for class_name in class_names:
    t = classifier_boost[class_name].predict(xgb.DMatrix(my_testing))
    pred.append(t)
pred = np.array(pred)
pred = pred.T

pred[:5]

# Transformed word - char vectors
with open('drive/My Drive/Colab Notebooks/sih2020/toxic-comment-challenge/models/xgboost.pickle', 'rb') as f:
  classifier_boost = pickle.load(f)

from xgboost import XGBClassifier

dir(XGBClassifier)

# preds = np.zeros((len(test), len(class_names)))
classifiers_boost = {}
for i, j in enumerate(class_names):
    m = XGBClassifier()
    print('fit', j)
    m.fit(train_features, y_tra[:, i])
    classifiers_boost[j] = m

model.summary()

pred = testing_predictions(classifiers_boost)

# sentence = x_val
# char_sent = char_vectorizer.transform(sentence)
# word_sent = word_vectorizer.transform(sentence)
# my_testing = hstack([char_sent, word_sent])
# pred = []
# for class_name in class_names:
#     t = classifiers[class_name].predict(my_testing)
#     pred.append(t)
# pred = np.array(pred)
# pred = pred.T

# return pred
# sum(pred)
print(classification_report(y_val, pred))

accuracy_score(y_val, pred)

