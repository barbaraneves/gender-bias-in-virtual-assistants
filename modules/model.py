import pandas as pd
import numpy as np
import os
from tqdm import *
import matplotlib.pyplot as plt

import tensorflow as tf
from transformers import TFBertForSequenceClassification

import time
from datetime import datetime, timezone, timedelta
from time import strftime, gmtime
from pathlib import Path
from contextlib import redirect_stdout

DEFAULT_FONT_SIZE = 14
plt.rcParams['font.size'] = DEFAULT_FONT_SIZE

# ---- Configuring use of TPU -----
use_tpu = True

if use_tpu:
    assert 'COLAB_TPU_ADDR' in os.environ, 'Missing TPU; did you request a TPU in Notebook Settings?'

if 'COLAB_TPU_ADDR' in os.environ:
  TF_MASTER = 'grpc://{}'.format(os.environ['COLAB_TPU_ADDR'])
else:
  TF_MASTER=''
  
tpu_address = TF_MASTER

resolver = tf.distribute.cluster_resolver.TPUClusterResolver(TF_MASTER)
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.experimental.TPUStrategy(resolver)

# ---- End of configuring use of TPU -----
  
def bert_linear_classifier(source_length, labels_size, metric):
    encoder_inputs_1 = tf.keras.Input(shape=(source_length), dtype='int32')
    encoder_inputs_2 = tf.keras.Input(shape=(source_length), dtype='int32')
    
    bert_encoder = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
    
    output = bert_encoder([encoder_inputs_1, encoder_inputs_2], training=True)
    out = tf.keras.layers.Dense(labels_size, activation=tf.nn.sigmoid)(output[0])
    
    model = tf.keras.Model(inputs=[encoder_inputs_1, encoder_inputs_2], outputs=[out])  
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5), loss=tf.keras.losses.binary_crossentropy, metrics=[metric])
    
    return model

def bert_linear_classifier2(source_length, hidden_units, labels_size, metric):
    with strategy.scope():
        encoder_inputs_1 = tf.keras.Input(shape=(source_length), dtype='int32')
        encoder_inputs_2 = tf.keras.Input(shape=(source_length), dtype='int32')
        
        bert_encoder = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
        bert_encoder_inputs = bert_encoder([encoder_inputs_1, encoder_inputs_2], training=True)
        
        output = tf.keras.layers.Dense(hidden_units, activation='relu')(bert_encoder_inputs[0])
        
        out = tf.keras.layers.Dense(labels_size, activation=tf.nn.sigmoid)(output)
        # drop = tf.keras.layers.Dropout(0.1)(out)
        
        model = tf.keras.Model(inputs=[encoder_inputs_1, encoder_inputs_2], outputs=[out]) # drop

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5), loss=tf.keras.losses.binary_crossentropy, metrics=[metric])
    
    return model

def bert_lstm_classifier(source_length, hidden_units, labels_size, dropout, metric):
    encoder_inputs_1 = tf.keras.Input(shape=(source_length), dtype='int32')
    encoder_inputs_2 = tf.keras.Input(shape=(source_length), dtype='int32')
    
    bert_encoder = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')
    bert_encoder_inputs = bert_encoder([encoder_inputs_1, encoder_inputs_2], training=True)
    
    pooled_input = tf.keras.layers.GlobalMaxPool1D()(bert_encoder_inputs[0])
    rnn_output = tf.keras.layers.LSTM(hidden_units, return_sequences=True, recurrent_dropout=dropout)(pooled_input)
    pooled_out = tf.keras.layers.GlobalMaxPool1D()(rnn_output)
    
    out = tf.keras.layers.Dense(labels_size, activation=tf.nn.sigmoid)(pooled_out)
    
    model = tf.keras.Model(inputs=[encoder_inputs_1, encoder_inputs_2], outputs=[out])  
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5), loss=tf.keras.losses.binary_crossentropy, metrics=[metric])
    
    return model

def lstm_classifier(source_length, hidden_units, vocab_size, labels_size, embedding_dim, embedding_matrix, metric):
    encoder_inputs = tf.keras.Input(shape=(source_length,), dtype='int32')
    
    embedding_layer = tf.keras.layers.Embedding(vocab_size,
                                                embedding_dim,
                                                weights=[embedding_matrix],
                                                input_length=source_length,
                                                trainable=False)
    embedded_sequences = embedding_layer(encoder_inputs)
    
    lstm_encoder = tf.keras.layers.LSTM(hidden_units, return_sequences=True)(embedded_sequences)
    pooled_out = tf.keras.layers.GlobalMaxPool1D()(lstm_encoder)
    drop1 = tf.keras.layers.Dropout(0.01)(pooled_out)
    
    output = tf.keras.layers.Dense(hidden_units, activation='relu')(drop1)
    drop2 = tf.keras.layers.Dropout(0.01)(output)
    
    out = tf.keras.layers.Dense(labels_size, activation=tf.nn.sigmoid)(drop2)
    model = tf.keras.Model(inputs=encoder_inputs, outputs=out)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5), loss=tf.keras.losses.binary_crossentropy, metrics=[metric])
    
    return model

class time_history(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []
    
    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()
    
    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

def run_text_classifier(model, X_train, y_train, X_dev, y_dev, batch_size, max_epochs, patience, eval_metric, monitor, source_length, hidden_units, labels_size, model_path, vocab_size=None, embedding_dim=None, embedding_matrix=None):
    if vocab_size:
        model = model(source_length, hidden_units, vocab_size, labels_size, embedding_dim, embedding_matrix, eval_metric)
    else: 
        model = model(source_length, hidden_units, labels_size, eval_metric)
    print(model.summary())

    time_callback = time_history()
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, min_delta=0.001, mode='max')
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, monitor=monitor, verbose=1, save_best_only=True, mode="max", save_weights_only=True)

    start_run_datetime = datetime.now().astimezone(timezone(timedelta(hours=-3))).strftime('%d/%m/%Y %H:%M')
    history = model.fit(X_train, 
                        y_train, 
                        validation_data=(X_dev, y_dev), 
                        batch_size=32, 
                        epochs=max_epochs, 
                        verbose=1, 
                        callbacks=[time_callback, checkpoint, early_stopping])
    run_time = strftime("%H:%M:%S", gmtime(sum(time_callback.times)))
    final_run_datetime = datetime.now().astimezone(timezone(timedelta(hours=-3))).strftime('%d/%m/%Y %H:%M')
    
    model_meta = {
        'model': model, 
        'history': history, 
        'start': start_run_datetime, 
        'run_time': run_time, 
        'final': final_run_datetime        
    }

    return model_meta
    
def plot_model_loss_score(history, score_name):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 10))
    
    ax0.plot(history.history[score_name], linewidth=3, color='darksalmon', marker='o')
    ax0.plot(history.history['val_' + score_name], linewidth=3, color='skyblue', marker='o')
    ax0.set_title('Score Values')
    ax0.set_ylabel('Scores')
    ax0.set_xlabel('Epoch')
    ax0.legend(['Treino', 'Desenvolvimento'], loc='upper left')
    
    ax1.plot(history.history['loss'], linewidth=3, color='darksalmon', marker='o')
    ax1.plot(history.history['val_loss'], linewidth=3, color='skyblue', marker='o')
    ax1.set_title('Loss Values')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Treino', 'Desenvolvimento'], loc='upper left')
    plt.show()
    
def change_target_codification(y, idx2labels, is_safe=False, pred_gender=[]):
    y_coded = []
    
    for index, labels in enumerate(y):
        labels_aux = [''] * len(labels)
        
        labeled_idx = [i for i, e in enumerate(labels.tolist()) if e == 1]
        safe_idx = [i for i, e in enumerate(labels.tolist()) if e == 0]
        
        if is_safe:
            for i in labeled_idx:
                labels_aux[i] = idx2labels[i]
            for j in safe_idx:
                labels_aux[j] = 'safe'
            
            if index == 0:    
                y_coded = labels_aux
            else:
                y_coded.extend(labels_aux)
        else:
            for i in labeled_idx:
                y_coded.append(idx2labels[i])
    return y_coded
    
def encode(count, label):
    encoded = np.zeros(count)
    encoded[int(label)] = 1
    return encoded
  
def label_encoder(y):
    count = int(y.max() + 1)
    return np.array([ encode(count, label) for label in y ])
  
def label_decoder(Y):  
    return Y.argmax(axis=1)
