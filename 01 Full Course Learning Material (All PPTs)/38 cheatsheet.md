################################################################################

# ML/DL COMPLETE MODELS CHEATSHEET

# 1 to 5 => Lenear Regession

# 6 to 10 => classification models

# 11 to => Neural Network

#

# Every model from syllabus with:

# - Imports

# - Model definition

# - Training pipeline

# - Evaluation

# - Prediction

#

# NO PRINT STATEMENTS - Pure code for exams

# Compact format - Copy-paste ready

################################################################################

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # means 
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

################################################################################

# 1. LINEAR REGRESSION (UNIVARIATE)

################################################################################

# Imports

from sklearn.linear_model import LinearRegression

# Data preparation

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 5])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model

model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Evaluate

y_pred_lr = model_lr.predict(X_test)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

# Predict

prediction_lr = model_lr.predict([[6]])

################################################################################

# 2. LINEAR REGRESSION (MULTIVARIATE)

################################################################################

# Imports (same as above)

# Data (multiple features)

X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([3, 5, 7, 9, 11])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model

model_mlr = LinearRegression()
model_mlr.fit(X_train, y_train)

# Evaluate

y_pred_mlr = model_mlr.predict(X_test)
rmse_mlr = np.sqrt(mean_squared_error(y_test, y_pred_mlr))
r2_mlr = r2_score(y_test, y_pred_mlr)

# Predict

prediction_mlr = model_mlr.predict([[6, 7]])

################################################################################

# 3. POLYNOMIAL REGRESSION

################################################################################

# Imports

from sklearn.preprocessing import PolynomialFeatures

# Data

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature engineering

poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Model

model_poly = LinearRegression()
model_poly.fit(X_train_poly, y_train)

# Evaluate

y_pred_poly = model_poly.predict(X_test_poly)
rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
r2_poly = r2_score(y_test, y_pred_poly)

# Predict

X_new = np.array([[6]])
X_new_poly = poly.transform(X_new)
prediction_poly = model_poly.predict(X_new_poly)

################################################################################

# 4. LOGISTIC REGRESSION (BINARY CLASSIFICATION)

################################################################################

# Imports

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# Data

X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model

model_log = LogisticRegression()
model_log.fit(X_train_scaled, y_train)

# Evaluate

y_pred_log = model_log.predict(X_test_scaled)
acc_log = accuracy_score(y_test, y_pred_log)
auc_log = roc_auc_score(y_test, model_log.predict_proba(X_test_scaled)[:, 1])

# Predict

prediction_log = model_log.predict([[0, 0]])
probability_log = model_log.predict_proba([[0, 0]])

################################################################################

# 5. LOGISTIC REGRESSION (MULTICLASS)

################################################################################

# Imports

from sklearn.datasets import load_iris

# Data

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model

model_multilog = LogisticRegression(multi_class='softmax', max_iter=1000)
model_multilog.fit(X_train_scaled, y_train)

# Evaluate

y_pred_multilog = model_multilog.predict(X_test_scaled)
acc_multilog = accuracy_score(y_test, y_pred_multilog)

# Predict

prediction_multilog = model_multilog.predict([[5.1, 3.5, 1.4, 0.2]])
probabilities_multilog = model_multilog.predict_proba([[5.1, 3.5, 1.4, 0.2]])

################################################################################

# 6. SUPPORT VECTOR MACHINE (SVM)

################################################################################

# Imports

from sklearn.svm import SVC

# Data

X, y = make_classification(n_samples=100, n_features=2, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling (essential for SVM)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model

model_svm = SVC(kernel='rbf', probability=True)
model_svm.fit(X_train_scaled, y_train)

# Evaluate

y_pred_svm = model_svm.predict(X_test_scaled)
acc_svm = accuracy_score(y_test, y_pred_svm)

# Predict

prediction_svm = model_svm.predict([[0, 0]])

################################################################################

# 7. DECISION TREE

################################################################################

# Imports

from sklearn.tree import DecisionTreeClassifier

# Data

X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model

model_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
model_dt.fit(X_train, y_train)

# Evaluate

y_pred_dt = model_dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)

# Predict

prediction_dt = model_dt.predict([[0.5, 0.5, 0.5, 0.5]])

################################################################################

# 8. RANDOM FOREST (BAGGING)

################################################################################

# Imports

from sklearn.ensemble import RandomForestClassifier

# Data

X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model

model_rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model_rf.fit(X_train, y_train)

# Evaluate

y_pred_rf = model_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

# Predict

prediction_rf = model_rf.predict([[0.5, 0.5, 0.5, 0.5]])

################################################################################

# 9. GRADIENT BOOSTING

################################################################################

# Imports

from sklearn.ensemble import GradientBoostingClassifier

# Data

X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model

model_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model_gb.fit(X_train, y_train)

# Evaluate

y_pred_gb = model_gb.predict(X_test)
acc_gb = accuracy_score(y_test, y_pred_gb)

# Predict

prediction_gb = model_gb.predict([[0.5, 0.5, 0.5, 0.5]])

################################################################################

# 10. XGBOOST

################################################################################

# Imports

import xgboost as xgb

# Data

X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model

model_xgb = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
model_xgb.fit(X_train, y_train)

# Evaluate

y_pred_xgb = model_xgb.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)

# Predict

prediction_xgb = model_xgb.predict([[0.5, 0.5, 0.5, 0.5]])

################################################################################

# 11. MULTILAYER PERCEPTRON (MLP) - NEURAL NETWORK

################################################################################

# Imports (TensorFlow/Keras)

# Data

X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model

model_mlp = keras.Sequential([
layers.Dense(64, activation='relu', input_shape=(10,)),
layers.Dropout(0.3),
layers.Dense(32, activation='relu'),
layers.Dropout(0.3),
layers.Dense(1, activation='sigmoid')
])

# Compile

model_mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train

history_mlp = model_mlp.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

# Evaluate

loss_mlp, acc_mlp = model_mlp.evaluate(X_test_scaled, y_test, verbose=0)

# Predict

y_pred_mlp = model_mlp.predict(X_test_scaled, verbose=0)
y_pred_mlp_class = (y_pred_mlp > 0.5).astype(int).flatten()

################################################################################

# 12. CONVOLUTIONAL NEURAL NETWORK (CNN) - 2D (IMAGE CLASSIFICATION)

################################################################################

# Imports (TensorFlow/Keras)

from tensorflow.keras.datasets import cifar10

# Data

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Model

model_cnn = keras.Sequential([
layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
layers.MaxPooling2D((2, 2)),
layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
layers.MaxPooling2D((2, 2)),
layers.Flatten(),
layers.Dense(128, activation='relu'),
layers.Dropout(0.5),
layers.Dense(10, activation='softmax')
])

# Compile

model_cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train

history_cnn = model_cnn.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2, verbose=0)

# Evaluate

loss_cnn, acc_cnn = model_cnn.evaluate(X_test, y_test, verbose=0)

# Predict

y_pred_cnn = model_cnn.predict(X_test[:5], verbose=0)
y_pred_cnn_class = np.argmax(y_pred_cnn, axis=1)

################################################################################

# 13. CONVOLUTIONAL NEURAL NETWORK (CNN) - 1D (TEXT CLASSIFICATION)

################################################################################

# Imports (TensorFlow/Keras)

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Data

vocab_size = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)
X_train = pad_sequences(X_train, maxlen=100)
X_test = pad_sequences(X_test, maxlen=100)

# Model

model_cnn1d = keras.Sequential([
layers.Embedding(vocab_size, 128, input_length=100),
layers.Conv1D(32, kernel_size=3, activation='relu'),
layers.GlobalMaxPooling1D(),
layers.Dense(64, activation='relu'),
layers.Dropout(0.3),
layers.Dense(1, activation='sigmoid')
])

# Compile

model_cnn1d.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train

history_cnn1d = model_cnn1d.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2, verbose=0)

# Evaluate

loss_cnn1d, acc_cnn1d = model_cnn1d.evaluate(X_test, y_test, verbose=0)

# Predict

y_pred_cnn1d = model_cnn1d.predict(X_test[:5], verbose=0)

################################################################################

# 14. RECURRENT NEURAL NETWORK (RNN) - BASIC

################################################################################

# Imports (TensorFlow/Keras)

# Data (sequence)

X = np.array([[[1], [2], [3]], [[2], [3], [4]], [[3], [4], [5]]])
y = np.array([[4], [5], [6]])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model

model_rnn = keras.Sequential([
layers.SimpleRNN(32, activation='relu', input_shape=(3, 1)),
layers.Dense(1)
])

# Compile

model_rnn.compile(optimizer='adam', loss='mse')

# Train

history_rnn = model_rnn.fit(X_train, y_train, epochs=20, batch_size=2, validation_split=0.2, verbose=0)

# Evaluate

loss_rnn = model_rnn.evaluate(X_test, y_test, verbose=0)

# Predict

y_pred_rnn = model_rnn.predict(X_test, verbose=0)

################################################################################

# 15. LONG SHORT-TERM MEMORY (LSTM)

################################################################################

# Imports (TensorFlow/Keras)

# Data (time-series)

from sklearn.preprocessing import MinMaxScaler
data = np.array([100, 102, 101, 103, 104, 105, 103, 106, 108, 110]).reshape(-1, 1)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

X_lstm = np.array([data_scaled[i:i+3] for i in range(len(data_scaled)-3)])
y_lstm = np.array([data_scaled[i+3] for i in range(len(data_scaled)-3)])

X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# Model

model_lstm = keras.Sequential([
layers.LSTM(50, activation='relu', input_shape=(3, 1)),
layers.Dense(1)
])

# Compile

model_lstm.compile(optimizer='adam', loss='mse')

# Train

history_lstm = model_lstm.fit(X_train, y_train, epochs=50, batch_size=2, validation_split=0.2, verbose=0)

# Evaluate

loss_lstm = model_lstm.evaluate(X_test, y_test, verbose=0)

# Predict

y_pred_lstm = model_lstm.predict(X_test, verbose=0)

################################################################################

# 16. GATED RECURRENT UNIT (GRU)

################################################################################

# Imports (TensorFlow/Keras)

# Data (same as LSTM)

X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# Model

model_gru = keras.Sequential([
layers.GRU(50, activation='relu', input_shape=(3, 1)),
layers.Dense(1)
])

# Compile

model_gru.compile(optimizer='adam', loss='mse')

# Train

history_gru = model_gru.fit(X_train, y_train, epochs=50, batch_size=2, validation_split=0.2, verbose=0)

# Evaluate

loss_gru = model_gru.evaluate(X_test, y_test, verbose=0)

# Predict

y_pred_gru = model_gru.predict(X_test, verbose=0)

################################################################################

# 17. BIDIRECTIONAL LSTM

################################################################################

# Imports (TensorFlow/Keras)

# Data

X_train, X_test, y_train, y_test = train_test_split(X_lstm, y_lstm, test_size=0.2, random_state=42)

# Model

model_bilstm = keras.Sequential([
layers.Bidirectional(layers.LSTM(50, activation='relu'), input_shape=(3, 1)),
layers.Dense(1)
])

# Compile

model_bilstm.compile(optimizer='adam', loss='mse')

# Train

history_bilstm = model_bilstm.fit(X_train, y_train, epochs=50, batch_size=2, validation_split=0.2, verbose=0)

# Evaluate

loss_bilstm = model_bilstm.evaluate(X_test, y_test, verbose=0)

# Predict

y_pred_bilstm = model_bilstm.predict(X_test, verbose=0)

################################################################################

# 18. SEQUENCE-TO-SEQUENCE WITH ATTENTION (Encoder-Decoder)

################################################################################

# Imports (TensorFlow/Keras)

# Data (simplified)

encoder_input = np.random.rand(10, 5, 10)
decoder_input = np.random.rand(10, 5, 10)
target = np.random.rand(10, 5, 10)

# Encoder

encoder_input_layer = keras.Input(shape=(5, 10))
encoder = layers.LSTM(32, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_input_layer)
encoder_states = [state_h, state_c]

# Decoder

decoder*input_layer = keras.Input(shape=(5, 10))
decoder_lstm = layers.LSTM(32, return_sequences=True, return_state=True)
decoder_outputs, *, \_ = decoder_lstm(decoder_input_layer, initial_state=encoder_states)

# Dense output

decoder_dense = layers.Dense(10, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Model

model_seq2seq = keras.Model([encoder_input_layer, decoder_input_layer], decoder_outputs)

# Compile

model_seq2seq.compile(optimizer='adam', loss='mse')

# Train

history_seq2seq = model_seq2seq.fit([encoder_input, decoder_input], target, epochs=10, batch_size=2, verbose=0)

# Predict

y_pred_seq2seq = model_seq2seq.predict([encoder_input[:2], decoder_input[:2]], verbose=0)

################################################################################

# 19. WORD EMBEDDINGS - WORD2VEC (CBOW)

################################################################################

# Imports

from gensim.models import Word2Vec

# Data

sentences = [['the', 'cat', 'sat', 'on', 'the', 'mat'],
             ['the', 'dog', 'ran', 'in', 'the', 'park']]

# Model

model_w2v = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, sg=0)

# Get embedding

embedding = model_w2v.wv['cat']

# Similar words

similar = model_w2v.wv.most_similar('cat', topn=3)

################################################################################

# 20. WORD EMBEDDINGS - SKIP-GRAM

################################################################################

# Imports (same as Word2Vec)

# Data

sentences = [['the', 'cat', 'sat', 'on', 'the', 'mat'],
             ['the', 'dog', 'ran', 'in', 'the', 'park']]

# Model (sg=1 for Skip-gram)

model_skipgram = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, sg=1)

# Get embedding

embedding_sg = model_skipgram.wv['dog']

# Similar words

similar_sg = model_skipgram.wv.most_similar('dog', topn=3)

################################################################################

# 21. NAIVE BAYES (Text Classification)
################################################################################
# Imports
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

# Data

texts = ['good movie', 'bad movie', 'excellent film', 'terrible acting']
labels = [1, 0, 1, 0]

# Vectorization

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(texts)
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, labels, test_size=0.2, random_state=42)

# Model

model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)

# Evaluate

y_pred_nb = model_nb.predict(X_test)
acc_nb = accuracy_score(y_test, y_pred_nb)

# Predict

text_new = vectorizer.transform(['good film'])
prediction_nb = model_nb.predict(text_new)

################################################################################

# 22. K-MEANS CLUSTERING (Unsupervised)

################################################################################
# Imports
from sklearn.cluster import KMeans

# Data

X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
# Model

model_kmeans = KMeans(n_clusters=2, random_state=42)
clusters = model_kmeans.fit_predict(X)

# Evaluate

inertia = model*kmeans.inertia*

# Predict
new_point = np.array([[1, 1]])
prediction_kmeans = model_kmeans.predict(new_point)

################################################################################

# 23. PRINCIPAL COMPONENT ANALYSIS (PCA)

################################################################################
# Imports
from sklearn.decomposition import PCA

# Data
X = np.random.rand(100, 10)

# Model
model_pca = PCA(n_components=2)
X_pca = model_pca.fit_transform(X)

# Evaluate
explained*variance = model_pca.explained_variance_ratio*

# Transform new data

X_new = np.random.rand(5, 10)
X_new_pca = model_pca.transform(X_new)

################################################################################

# 24. HOUSE PRICE PREDICTION - LINEAR REGRESSION (EXAM TEMPLATE)

################################################################################

# Imports
import pandas as pd
from sklearn.linear_model import LinearRegression

# Data (area, location_encoded)
data = pd.DataFrame({
'area': [1000, 1500, 2000, 2500, 3000],
'location': [0, 1, 0, 1, 0],
'price': [200000, 300000, 400000, 500000, 600000]
})

X = data[['area', 'location']]
y = data['price']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model_house = LinearRegression()
model_house.fit(X_train, y_train)

# Evaluate
y_pred_house = model_house.predict(X_test)
rmse_house = np.sqrt(mean_squared_error(y_test, y_pred_house))
r2_house = r2_score(y_test, y_pred_house)

# Predict
price_prediction = model_house.predict([[2200, 1]])

################################################################################

# 25. IMAGE CLASSIFICATION - TRANSFER LEARNING (VGG16)
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load pre-trained model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

# Custom top layers
model_transfer = keras.Sequential([
base_model,
layers.Flatten(),
layers.Dense(256, activation='relu'),
layers.Dropout(0.5),
layers.Dense(10, activation='softmax')
])

# Compile
model_transfer.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# You would train with your own data:
# history = model_transfer.fit(train_data, validation_data=val_data, epochs=10)
# Predict on new image
# image = load_image('image.jpg')
# prediction = model_transfer.predict(image)

################################################################################

# 26. TEXT SENTIMENT ANALYSIS - LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Data
texts = ['great movie', 'terrible film', 'amazing acting', 'bad performance']
sentiments = [1, 0, 1, 0]

# Tokenize
tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, maxlen=10)

# Model
model_sentiment = keras.Sequential([
layers.Embedding(100, 32, input_length=10),
layers.LSTM(64, activation='relu'),
layers.Dense(1, activation='sigmoid')
])

# Compile
model_sentiment.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
history_sentiment = model_sentiment.fit(padded, sentiments, epochs=10, batch_size=2, verbose=0)

# Predict
new_text = tokenizer.texts_to_sequences(['good film'])
new_padded = pad_sequences(new_text, maxlen=10)
prediction_sentiment = model_sentiment.predict(new_padded, verbose=0)

################################################################################

# 27. NAMED ENTITY RECOGNITION (NER) - BILSTM

################################################################################

# Imports (for production use gensim, spacy)

# Example simplified version:

# Data (word indices, tags)

X_ner = np.random.randint(0, 100, (10, 20))
y_ner = np.random.randint(0, 5, (10, 20))

# Model

model_ner = keras.Sequential([
layers.Embedding(100, 64, input_length=20),
layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
layers.Dense(5, activation='softmax')
])

# Compile

model_ner.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train

history_ner = model_ner.fit(X_ner, y_ner, epochs=10, batch_size=2, verbose=0)

################################################################################

# EVALUATION METRICS SUMMARY

################################################################################

# Regression

# - RMSE = sqrt(mean((y_true - y_pred)^2))

# - R² = 1 - (SS_res / SS_tot)

# - MAE = mean(|y_true - y_pred|)

# Classification

# - Accuracy = (TP + TN) / (TP + TN + FP + FN)

# - Precision = TP / (TP + FP)

# - Recall = TP / (TP + FN)

# - F1 = 2 _ (Precision _ Recall) / (Precision + Recall)

# - AUC-ROC = area under ROC curve

# RNN/Sequence

# - Perplexity = exp(-1/N \* sum(log(p(x_i))))

# - BLEU = geometric mean of n-gram precision

################################################################################

# QUICK REFERENCE - IMPORT ALL AT ONCE

################################################################################

"""

# Quick imports for exam

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from tensorflow.keras import Sequential, layers, models
from tensorflow.keras.datasets import cifar10, imdb
from gensim.models import Word2Vec
import xgboost as xgb
"""

################################################################################

# END OF CHEATSHEET

################################################################################
