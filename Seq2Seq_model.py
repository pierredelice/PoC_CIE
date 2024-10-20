import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from model_params import parameters

device = parameters.get('device')

# Step 3: Define the Model Classes
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout = 0.5):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.bigru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        x = self.dropout(x)
        outputs, hidden = self.bigru(x, h0)
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size * 3, hidden_size)  # Corrected to hidden_size * 3
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, encoder_outputs, hidden):
        batch_size, seq_len, _ = encoder_outputs.size()
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # Repeat hidden state
        # Concatenate hidden state and encoder outputs
        combined = torch.cat((encoder_outputs, hidden), dim=2)
        energy = torch.tanh(self.attention(combined))
        attention_weights = torch.softmax(self.v(energy), dim=1)
        context_vector = torch.sum(attention_weights * encoder_outputs, dim=1)
        return context_vector, attention_weights

class DecoderWithAttention(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers=1, dropout=0.5):
        super(DecoderWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size * 2 + output_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden, encoder_outputs):
        context_vector, attention_weights = self.attention(encoder_outputs, hidden[-1])
        x = torch.cat((x, context_vector.unsqueeze(1)), dim=2)
        x = self.dropout(x)
        output, hidden = self.gru(x, hidden)
        output = self.fc(output.squeeze(1))
        return output, hidden, attention_weights

class Seq2Seq(nn.Module):
    def __init__(self, input_size, embedding_size, output_size, 
                 hidden_size, num_layers=1, dropout = 0.5):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.encoder = Encoder(embedding_size, hidden_size, num_layers, dropout)
        self.decoder = DecoderWithAttention(output_size, hidden_size, num_layers, dropout)
    
    def forward(self, src, trg):
        embedded = self.embedding(src)
        encoder_outputs, hidden = self.encoder(embedded)
        
        # Initialize the hidden state for the decoder (use sum or just forward direction)
        hidden = hidden.view(self.encoder.num_layers, 2, -1, self.encoder.hidden_size)
        hidden = hidden.sum(dim=1)  # Sum the forward and backward hidden states
        
        # Initialize the input to the decoder
        output = torch.zeros((trg.size(0), 1, trg.size(2))).to(src.device)  # Start token
        output = output.to(device)
        outputs = []
        for t in range(trg.size(1)):
            output, hidden, _ = self.decoder(output, hidden, encoder_outputs)
            outputs.append(output.unsqueeze(1))
            # Use teacher forcing: feed the target as the next input
            if t < trg.size(1) - 1:
                output = trg[:, t].unsqueeze(1).float()
        outputs = torch.cat(outputs, dim=1)
        return outputs
