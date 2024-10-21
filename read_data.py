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
df = pd.read_pickle("Data/icd_clean.pkl").rename(columns={'label':'causa_icd'})


#Run spellchecker to clean data
spell = SpellChecker(language='es')
def correct_word(word):
  return spell.correction(word)

#df['cause'] = df['cause'].progress_apply(correct_word)

#Calculate vocabulary
label_mapping = {value: label for label, value in enumerate(df['causa_icd'].unique())}
df['label'] = df['causa_icd'].map(label_mapping)
text, label = df['cause'].values, df['label'].values

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

labels = torch.tensor(df['label'].astype('category').cat.codes.values)

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

list_of_tuples = list(zip(df['cause'], df['causa_icd']))
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

import spacy
from unidecode import unidecode
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from functools import lru_cache

nlp = spacy.load('es_core_news_md')

def clean_description(text):
  if not isinstance(text, str):
    return ""
  text = unidecode(text.lower().strip())
  punctuation = '''|!()-[]{};:'"\,<>./?@#$%^&*_~'''
  text = text.translate(str.maketrans('', '', punctuation))
  words = word_tokenize(text)
  spanish_stopwords = set(stopwords.words('spanish'))
  filtered_words = [word for word in words if word not in spanish_stopwords]
  filtered_text = ' '.join(filtered_words)
  doc = nlp(filtered_text)
  lemmatized_words = ' '.join(token.lemma_ for token in doc)
  
  return lemmatized_words

# Initialize SpellChecker once
spell = SpellChecker(language='es')
custom_words = {"covid", "sars", "cov", "sars-cov2", "covid19"}
spell.word_frequency.load_words(custom_words)

# Cache corrected words
@lru_cache(maxsize=10000)
def correct_word(word):
    if word in custom_words:
        return word
    return spell.correction(word) or word

def spellcheck_correction(text):
    words = text.split()
    corrected_words = [correct_word(word) for word in words]
    return ' '.join(corrected_words)

#Save model and results
import pickle, os

os.makedirs('results', exist_ok=True)

torch.save(model.state_dict(),
           'results/seq2seq_model.pth')
with open('results/tokenizer.pkl', 'wb') as f:
   pickle.dump({'word_to_idx': word_to_idx, 'label_encoder':label_encoder}, f)


def load_model_and_tokenizer(model_path, tokenizer_path):
    # Load the model
    model = Seq2Seq(input_size, embedding_size, output_size, hidden_dim)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    
    # Load the tokenizer and label encoder
    with open(tokenizer_path, 'rb') as f:
        tokenizer_data = pickle.load(f)
        word_to_idx = tokenizer_data['word_to_idx']
        label_encoder = tokenizer_data['label_encoder']
    
    return model, word_to_idx, label_encoder


loaded_model, loaded_word_to_idx, loaded_label_encoder = \
    load_model_and_tokenizer('results/seq2seq_model.pth', 
                             'results/tokenizer.pkl')

#Prediction
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Step 10: Make predictions using only diagnosis
small_data = [
    "1 82 chocar cardiogenico",
    "2 53 covid virus identificar",
    "2 33 hipertension arteriel",
    "2 53 sars cov",
    "2 40 chocar hipovolemico",
    "1 82 choque cardiogenico",
    "2 53 covid virus identificar",
    "2 52 acidosis metabólico",
    "1 57 infarto agudo miocardio",
    "1 63 insuficiencia respiratorio agudo",
    "1 49 hipertensión arterial",
    "2 79 insuficiencia renal crónico",
    "2 69 síndrome dificultad respiratorio adulto",
    "1 70 insuficiencia respiratorio",
    "2 64 choque séptico",
    "2 10 Juana ingreso y murio por problemas cardiacos",
    "2 65 Margarita murio de problemas cardiacos"
]


# Preprocess the small data
preprocessed_small_data = [spellcheck_correction(clean_description(diagnosis)) for diagnosis in small_data]

small_dataset = MedicalDataset(preprocessed_small_data, loaded_word_to_idx, loaded_label_encoder, is_prediction=True)
small_loader = DataLoader(small_dataset, batch_size=len(small_dataset), shuffle=False, 
                          collate_fn=lambda x: torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=loaded_word_to_idx['<PAD>']))

for src in small_loader:
    src = src.to(device)
    trg_onehot = torch.zeros((src.size(0), 1, output_size)).to(device)  
    
    outputs = loaded_model(src, trg_onehot)
    _, predicted = torch.max(outputs, dim=-1)
    
    predicted_labels = [loaded_label_encoder.inverse_transform([pred.item()])[0] for pred in predicted.flatten()]
    print("Predicted labels:", predicted_labels)


