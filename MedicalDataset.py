import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from torch.nn.utils.rnn import pad_sequence


# Step 1: Define the Dataset Class
class MedicalDataset(Dataset):
    def __init__(self, data, word_to_idx, label_encoder, is_prediction = False):
        self.data = data
        self.word_to_idx = word_to_idx
        self.label_encoder = label_encoder
        self.is_prediction = is_prediction
        self.preprocess()
        
    def preprocess(self):
        self.cause_sequences = []
        self.causa_icd_sequences = []
        
        for item in self.data:
            # For prediction, item contains only diagnosis text
            if self.is_prediction:
                cause = item
                causa_icd = None
            else:
                cause, causa_icd = item
            
            # Tokenize cause
            cause_tokens = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in cause.split()]
            self.cause_sequences.append(torch.tensor(cause_tokens, dtype=torch.long))
            
            # Encode causa_icd if available
            if causa_icd is not None:
                causa_icd_encoded = self.label_encoder.transform([causa_icd])[0]
                self.causa_icd_sequences.append(causa_icd_encoded)
    
    def __len__(self):
        return len(self.cause_sequences)
    
    def __getitem__(self, idx):
        if self.is_prediction:
            return self.cause_sequences[idx]
        else:
            return self.cause_sequences[idx], self.causa_icd_sequences[idx]


# Step 2: Tokenization and Label Encoding
def prepare_data(data):
    all_words = [word for cause, _ in data for word in cause.split()]
    word_counts = Counter(all_words)
    vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    word_to_idx = {word: idx + 2 for idx, word in enumerate(vocab)}  # Start indexing from 2
    word_to_idx['<PAD>'] = 0
    word_to_idx['<UNK>'] = 1

    # Encode labels
    label_encoder = LabelEncoder()
    labels = [causa_icd for _, causa_icd in data]
    label_encoder.fit(labels)
    
    return word_to_idx, label_encoder


# Step 6: Define the Collate Function
def collate_fn(batch, pad_idx):
    sources, targets = zip(*batch)
    sources_padded = pad_sequence(sources, batch_first=True, padding_value=pad_idx)
    targets = torch.tensor(targets, dtype=torch.long)
    return sources_padded, targets
