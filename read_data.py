import pandas as pd
import numpy as np
import json
from spellchecker import SpellChecker
from tqdm import tqdm
tqdm.pandas()
from pprint import pprint
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.optim as optim
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model_params import parameters
from Seq2Seq_model import Seq2Seq
from train_eval import train_and_evaluate_model


#Read data
df = pd.read_pickle("Data/icd_clean.pkl")

#Run spellchecker to clean data
spell = SpellChecker(language='es')
def correct_word(word):
  return spell.correction(word)

#df['cause'] = df['cause'].progress_apply(correct_word)

#Calculate vocabulary
label_mapping = {value: label for label, value in enumerate(df['label'].unique())}
df['labels'] = df['label'].map(label_mapping)
text, label = df['cause'].values, df['labels'].values

vocabulary = set([word for item in text for word in str(item).split()])

# Save vocabulary as a JSON file
vocab_list = list(vocabulary)
with open('Data/vocabulary.json', 'w') as f:
    json.dump(vocab_list, f)

#Tokenization
tokenizer = Tokenizer(num_words = parameters['vocab_size'],
                      oov_token = '<OOV>')
tokenizer.fit_on_texts(df['cause'])
sequences = tokenizer.texts_to_sequences(df['cause'])
padded_sequences = pad_sequences(sequences, 
                                 maxlen=parameters['max_length'],
                                 padding='post'
                                 )

labels = torch.tensor(df['labels'].astype('category').cat.codes.values)

#Training and testing data
from sklearn.model_selection import train_test_split as tts
X_train, X_test, y_train, y_test = tts(padded_sequences, labels,
                                       test_size = parameters['test_size'],
                                       random_state = parameters['random_state']
                                       )
#Dataloader 
from torch.utils.data import Dataset, DataLoader, TensorDataset
X_train = torch.tensor(X_train, dtype=torch.long).to(parameters['device'])
X_test = torch.tensor(X_test, dtype=torch.long).to(parameters['device'])
y_train = torch.tensor(y_train, dtype=torch.long).to(parameters['device'])
y_test = torch.tensor(y_test, dtype=torch.long).to(parameters['device'])

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, 
                          batch_size=parameters['batch_size'],
                          shuffle=True)
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, 
                         batch_size=parameters['batch_size'],
                         shuffle=False)


#Applied MedicalDataset function
from MedicalDataset import prepare_data, MedicalDataset, collate_fn
from torch.utils.data import random_split 

list_of_tuples = list(zip(df['cause'], df['labels']))
word_to_idx, label_encoder, vocab = prepare_data(list_of_tuples)
dataset = MedicalDataset(list_of_tuples, word_to_idx, label_encoder)

#Use train test split 
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
device = parameters.get('device')

train_loader = DataLoader(train_dataset,
                          batch_size = parameters['batch_size'],
                          shuffle = True,
                          collate_fn=lambda x: collate_fn(x, word_to_idx['<PAD>'])
                          )
test_loader = DataLoader(test_dataset, 
                         batch_size = parameters['batch_size'],
                         shuffle=False,
                         collate_fn = lambda x: collate_fn(x, word_to_idx['<PAD>'])
                         )
input_size = len(word_to_idx)
embedding_size = 512#parameters['embedding_dim']
output_size = len(label_encoder.classes_)
hidden_dim = 128

from sklearn.utils.class_weight import compute_class_weight

dataset = train_dataset.dataset
indices = train_dataset.indices

# Initialize an empty list to store labels
labels = []

# Loop through the indices and extract labels from the MedicalDataset
for idx in indices:
    _, label = dataset[idx]  # Assuming dataset[idx] returns (data, label)
    labels.append(label)

# Convert the labels list to a numpy array
labels = np.array(labels)

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(labels),
                                     y=labels)

# Convert to torch tensor if needed for model usage
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)


# class_weights = compute_class_weight(class_weight='balanced',
#                                      classes=np.unique(train_dataset.tensors[1].cpu().numpy()),
#                                      y=train_dataset.tensors[1].cpu().numpy())
# class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
del X_train, y_train, X_test, y_test, train_dataset, test_dataset, padded_sequences

criterion = nn.CrossEntropyLoss(weight = class_weights)
model = Seq2Seq(input_size, embedding_size, output_size, hidden_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# Train and evaluate the model
num_epochs = 3
training_losses, training_accuracy, evaluation_accuracy, evaluation_precision, evaluation_recall, evaluation_f1 = train_and_evaluate_model(
    model, train_loader, test_loader, criterion, optimizer, num_epochs
)
