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

# Step 4: Define Training and Evaluation Function
def train_and_evaluate_model(model, train_loader, test_loader, 
                             criterion, optimizer, num_epochs, patience = 3):
    model.train()
    
    # Lists to store metrics for each epoch
    training_losses = []
    training_accuracy = []
    evaluation_accuracy = []
    evaluation_precision = []
    evaluation_recall = []
    evaluation_f1 = []
    output_size = 20

    best_f1 = 0  # Best F1 score for early stopping
    epochs_without_improvement = 0  # Count of epochs without improvement
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total = 0
        correct = 0
        epoch_loss = 0
        
        for src, trg in train_loader:
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            
            # One-hot encode trg for decoder
            trg_onehot = torch.nn.functional.one_hot(trg, num_classes=output_size).float().to(device)
            trg_onehot = trg_onehot.unsqueeze(1) if trg_onehot.dim() == 2 else trg_onehot
            
            outputs = model(src, trg_onehot)
            
            # Compute loss
            trg_flat = trg.view(-1)
            outputs_flat = outputs.view(-1, output_size)
            outputs_flat = outputs_flat.to(device)  # Move outputs to MPS
            trg_flat = trg_flat.to(device)          # Move targets to MPS

            loss = criterion(outputs_flat, trg_flat)
            loss.backward()
            optimizer.step()
            
            # Calculate training accuracy
            epoch_loss += loss.item() * trg.size(0)
            _, predicted = torch.max(outputs_flat, 1)
            total += trg_flat.size(0)
            correct += (predicted == trg_flat).sum().item()
        
        # Average loss and accuracy for the epoch
        epoch_loss /= total
        epoch_accuracy = 100 * correct / total
        training_losses.append(epoch_loss)
        training_accuracy.append(epoch_accuracy)
        
        # Evaluation phase
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for src, trg in test_loader:
                src, trg = src.to(device), trg.to(device)
                trg_onehot = torch.nn.functional.one_hot(trg, num_classes=output_size).float().to(device)
                trg_onehot = trg_onehot.unsqueeze(1) if trg_onehot.dim() == 2 else trg_onehot
                
                outputs = model(src, trg_onehot)
                _, predicted = torch.max(outputs, dim=-1)
                
                all_preds.extend(predicted.cpu().numpy().flatten())
                all_labels.extend(trg.cpu().numpy().flatten())
        
        # Calculate evaluation metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        
        evaluation_accuracy.append(accuracy)
        evaluation_precision.append(precision)
        evaluation_recall.append(recall)
        evaluation_f1.append(f1)
        
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.2f}, Training Accuracy: {epoch_accuracy:.2f}, '
              f'Evaluation Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1:.2f}')
        
        #Early stopping check
        if f1 > best_f1:
            best_f1 = f1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break
    
    return training_losses, training_accuracy, evaluation_accuracy, evaluation_precision, evaluation_recall, evaluation_f1