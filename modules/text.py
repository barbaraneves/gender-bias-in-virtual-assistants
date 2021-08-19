import pandas as pd
import numpy as np
from tqdm import *

from transformers import BertTokenizer
from keras.preprocessing.sequence import pad_sequences
from skmultilearn.model_selection import iterative_train_test_split

MAX_LENGTH = 60

def remove_potencial_stopwords(tokens, words2ignore):
    return [word for word in tokens if word not in words2ignore]
    
def padding_sequences(texts, max_length):
    tokens_ids = []
    masked_ids = []
    
    print("Downloading BERT's tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    with tqdm_notebook(total=len(texts), desc='\nText to sequences for BERT Classifier') as pbar:
        for text in texts:
            encoding = tokenizer.encode_plus(text)
            input_ids, attention_id = encoding["input_ids"], encoding["attention_mask"] 
            
            tokens_ids.append(input_ids)
            masked_ids.append(attention_id)
            pbar.update(1)
    pbar.close()
    
    print('Padding the sequences...')
    padded_tokens_ids = pad_sequences(tokens_ids, maxlen=max_length, padding='post', value=0)
    padded_masked_ids = pad_sequences(masked_ids, maxlen=max_length, padding='post', value=0)
    
    return padded_tokens_ids, padded_masked_ids
    
def padding_sequences_ftext(tokens, vocab, max_length):
    tokens_ids = []
    vocab_words = vocab.keys()
    
    with tqdm_notebook(total=len(tokens), desc='\nText to sequences for LSTM Classifier with FastText embeddings') as pbar:
        for tokens_list in tokens:
            input_ids = [vocab.get(token)[0] if vocab.get(token) else 0 for token in tokens_list]
            tokens_ids.append(input_ids)
            pbar.update(1)
    pbar.close()
    
    print('Padding the sequences...')
    padded_tokens_ids = pad_sequences(tokens_ids, maxlen=max_length, padding='post', value=0)
    return padded_tokens_ids
    
def data_split(X, y, test_size, dev_size, random_seed):
    assert (test_size >= 0 and dev_size <= 1) or (dev_size >= 0 and dev_size <= 1), "Indevid set fraction"
    
    np.random.seed(random_seed)
    X_remain, y_remain, X_test, y_test = iterative_train_test_split(X, y, test_size=test_size)
    
    np.random.seed(random_seed)
    X_train, y_train, X_dev, y_dev = iterative_train_test_split(X_remain, y_remain, test_size=dev_size)
    
    return X_train, y_train, X_test, y_test, X_dev, y_dev
