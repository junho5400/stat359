"""
Train an LSTM for sentiment classification using padded FastText word vectors.

Author: Junho
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import gensim.downloader as api
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import os

# ============================================================================
# 1. Setup and Reproducibility
# ============================================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Set device (CUDA > MPS > CPU)
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

# Hyperparameters
BATCH_SIZE = 32
EMBEDDING_DIM = 300      # FastText embedding dimension
HIDDEN_DIM = 128         # LSTM hidden state dimension
NUM_LAYERS = 2           # Number of LSTM layers
DROPOUT = 0.3            # Dropout rate
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
NUM_CLASSES = 3          # negative, neutral, positive
MAX_SEQ_LEN = 32         # Pad/truncate to exactly 32 tokens

# Create output directory
os.makedirs('outputs', exist_ok=True)

# ============================================================================
# 2. Load and preprocess dataset
# ============================================================================
print("\n========== Loading Dataset ==========")
dataset = load_dataset('financial_phrasebank', 'sentences_50agree', trust_remote_code=True)
print(f"Total samples: {len(dataset['train'])}")

# Split data into texts and labels
texts = dataset['train']['sentence']
labels = dataset['train']['label']

print(f"\nClass distribution:")
unique, counts = np.unique(labels, return_counts=True)
for label, count in zip(unique, counts):
    print(f"  Class {label}: {count}")

# ============================================================================
# 3. Train/Val/Test Split (Stratified)
# ============================================================================
print("\n========== Creating Splits ==========")

# Step 1: train+val (85%) vs test (15%)
train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
    texts, labels, 
    test_size=0.15, 
    random_state=SEED, 
    stratify=labels
)

# Step 2: train (85% of train+val) vs val (15% of train+val)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_val_texts, train_val_labels,
    test_size=0.15,
    random_state=SEED,
    stratify=train_val_labels
)

print(f"Train size: {len(train_texts)}")
print(f"Val size: {len(val_texts)}")
print(f"Test size: {len(test_texts)}")

# ============================================================================
# 4. Compute Class Weights (Handle Imbalanced Dataset)
# ============================================================================
print("\n========== Computing Class Weights ==========")
train_label_counts = np.bincount(train_labels)
total_samples = len(train_labels)
class_weights = total_samples / (NUM_CLASSES * train_label_counts)
class_weights = torch.FloatTensor(class_weights).to(device)
print(f"Class weights: {class_weights}")

# ============================================================================
# 5. Load FastText Embeddings
# ============================================================================
print("\n========== Loading FastText Embeddings ==========")
fasttext_model = api.load('fasttext-wiki-news-subwords-300')
print("FastText model loaded successfully!")

# ============================================================================
# 6. Sequence Embedding Function (Pad/Truncate to MAX_SEQ_LEN)
# ============================================================================
def get_padded_sequence(sentence, model, max_len=MAX_SEQ_LEN):
    """
    Tokenize sentence and create a padded/truncated sequence of word vectors.
    
    Args:
        sentence: Input sentence (string)
        model: FastText model
        max_len: Maximum sequence length (default: 32)
    
    Returns:
        Numpy array of shape (max_len, EMBEDDING_DIM)
        Each row is a 300-dim word vector; padded with zeros if needed
    """
    # Simple tokenization: lowercase and split by spaces
    tokens = sentence.lower().split()
    
    # Get word vectors for each token
    vectors = []
    for token in tokens[:max_len]:  # Truncate to max_len
        if token in model:
            vectors.append(model[token])
        else:
            # Use zero vector for unknown words
            vectors.append(np.zeros(EMBEDDING_DIM))
    
    # Pad with zero vectors if sequence is shorter than max_len
    while len(vectors) < max_len:
        vectors.append(np.zeros(EMBEDDING_DIM))
    
    return np.array(vectors)  # Shape: (max_len, EMBEDDING_DIM)

# ============================================================================
# 7. PyTorch Dataset Class
# ============================================================================
class LSTMSentimentDataset(Dataset):
    """
    Dataset for LSTM sentiment classification.
    Each sample is a padded sequence of word vectors and a label.
    """
    def __init__(self, texts, labels, fasttext_model, max_len=MAX_SEQ_LEN):
        self.texts = texts
        self.labels = labels
        self.fasttext_model = fasttext_model
        self.max_len = max_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Get padded sequence of word vectors (shape: max_len x EMBEDDING_DIM)
        sequence = get_padded_sequence(text, self.fasttext_model, self.max_len)
        
        return torch.FloatTensor(sequence), torch.LongTensor([label])[0]

# Create datasets
print("\n========== Creating Datasets ==========")
train_dataset = LSTMSentimentDataset(train_texts, train_labels, fasttext_model)
val_dataset = LSTMSentimentDataset(val_texts, val_labels, fasttext_model)
test_dataset = LSTMSentimentDataset(test_texts, test_labels, fasttext_model)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# ============================================================================
# 8. Define LSTM Model
# ============================================================================
class LSTMClassifier(nn.Module):
    """
    LSTM-based sentiment classifier.
    
    Architecture:
    Input: (batch_size, seq_len, embedding_dim)
    LSTM: processes sequence and outputs hidden states
    Classification: use final hidden state to predict sentiment
    """
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_classes, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM layer
        # batch_first=True means input/output tensors are (batch, seq, feature)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,  # Dropout between LSTM layers
            batch_first=True
        )
        
        # Dropout layer (applied to final LSTM output)
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, embedding_dim)
        
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # LSTM forward pass
        # lstm_out shape: (batch_size, seq_len, hidden_dim)
        # h_n shape: (num_layers, batch_size, hidden_dim)
        # c_n shape: (num_layers, batch_size, hidden_dim)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the final hidden state from the last LSTM layer
        # h_n[-1] has shape (batch_size, hidden_dim)
        final_hidden = h_n[-1]
        
        # Apply dropout
        out = self.dropout(final_hidden)
        
        # Classify
        logits = self.fc(out)
        
        return logits

# Initialize model
model = LSTMClassifier(
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    dropout=DROPOUT
).to(device)

print("\n========== Model Architecture ==========")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

# ============================================================================
# 9. Loss and Optimizer Setup
# ============================================================================
# Weighted CrossEntropyLoss to handle class imbalance
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)

# ============================================================================
# 10. Training and Evaluation Functions
# ============================================================================
def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for sequences, labels in tqdm(loader, desc="Training"):
        sequences, labels = sequences.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Store predictions for metrics
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, accuracy, macro_f1

def evaluate(model, loader, criterion, device):
    """Evaluate on validation or test set"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in tqdm(loader, desc="Evaluating"):
            sequences, labels = sequences.to(device), labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, accuracy, macro_f1, all_preds, all_labels

# ============================================================================
# 11. Training Loop
# ============================================================================
print("\n========== Starting Training ==========")

# Lists to store metrics
train_losses, train_accs, train_f1s = [], [], []
val_losses, val_accs, val_f1s = [], [], []

best_val_f1 = 0.0
best_model_path = 'outputs/best_lstm_model.pt'

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    
    # Train
    train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    train_f1s.append(train_f1)
    
    # Validate
    val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    val_f1s.append(val_f1)
    
    # Learning rate scheduling
    scheduler.step(val_f1)
    
    print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
    
    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), best_model_path)
        print(f"✓ New best model saved! (Val F1: {val_f1:.4f})")

# ============================================================================
# 12. Final Test Set Evaluation
# ============================================================================
print("\n========== Testing Best Model ==========")
model.load_state_dict(torch.load(best_model_path))
test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
    model, test_loader, criterion, device
)

print(f"\nTest Results:")
print(f"  Loss: {test_loss:.4f}")
print(f"  Accuracy: {test_acc:.4f}")
print(f"  Macro F1: {test_f1:.4f}")

# Check if goal is achieved
if test_f1 >= 0.70:
    print(f"\n✓ SUCCESS: Test Macro F1 ({test_f1:.4f}) >= 0.70")
else:
    print(f"\n✗ WARNING: Test Macro F1 ({test_f1:.4f}) < 0.70")

# ============================================================================
# 13. Visualization: Training Curves
# ============================================================================
print("\n========== Generating Plots ==========")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Loss curve
axes[0].plot(train_losses, label='Train Loss')
axes[0].plot(val_losses, label='Val Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('LSTM: Loss vs Epochs')
axes[0].legend()
axes[0].grid(True)

# Accuracy curve
axes[1].plot(train_accs, label='Train Accuracy')
axes[1].plot(val_accs, label='Val Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('LSTM: Accuracy vs Epochs')
axes[1].legend()
axes[1].grid(True)

# F1 curve
axes[2].plot(train_f1s, label='Train Macro F1')
axes[2].plot(val_f1s, label='Val Macro F1')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Macro F1 Score')
axes[2].set_title('LSTM: Macro F1 vs Epochs')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig('outputs/lstm_training_curves.png', dpi=300, bbox_inches='tight')
print("Saved: outputs/lstm_training_curves.png")

# ============================================================================
# 14. Visualization: Confusion Matrix
# ============================================================================
cm = confusion_matrix(test_labels, test_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'LSTM Confusion Matrix (Test F1: {test_f1:.4f})')
plt.savefig('outputs/lstm_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Saved: outputs/lstm_confusion_matrix.png")

print("\n========== Training Complete! ==========")