import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 512  # change it to fit your memory constraints, e.g., 256, 128 if you run out of memory
EPOCHS = 5
LEARNING_RATE = 0.001
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    def __init__(self, skipgram_df):
        """
        Args:
            skipgram_df: DataFrame with 'center' and 'context' columns
        """
        self.center = torch.tensor(skipgram_df['center'].values, dtype=torch.long)
        self.context = torch.tensor(skipgram_df['context'].values, dtype=torch.long)

    def __len__(self):
        return len(self.center)
    
    def __getitem__(self, idx):
        return self.center[idx], self.context[idx]

# Simple Skip-gram Module
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
        """
        super(Word2Vec, self).__init__()
        
        # Input embeddings (center words)
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Output embeddings (context words)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Initialize embeddings
        self.center_embeddings.weight.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)
        self.context_embeddings.weight.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)
    
    def forward(self, center_words, context_words):
        """
        Compute dot product between center and context embeddings

        Args:
            center_words: [batch_size] or [batch_size, 1]
            context_words: [batch_size, num_words] where num_words can be 1 or NEGATIVE_SAMPLES+1

        Returns:
            scores: [batch_size, num_words]
        """
        # Get embeddings
        center_embeds = self.center_embeddings(center_words) # [batch_size, embedding_dim]
        context_embeds = self.context_embeddings(context_words) # [batch_size, num_words, embedding_dim]
        
        # Handle different dimensions
        if center_embeds.dim() == 2 and context_embeds.dim() == 3:
            # Compute dot product: [batch_size, 1, embedding_dim] @ [batch_size, embedding_dim, num_words]
            scores = torch.bmm(context_embeds, center_embeds.unsqueeze(2)).squeeze(2) # [batch_size, num_words]
        else:
            # Fallback for 2D case
            scores = torch.sum(center_embeds * context_embeds, dim=-1)
        
        return scores
    
    def get_embeddings(self):
        """Return the trained center embeddings as numpy array"""
        return self.center_embeddings.weight.data.cpu().numpy()


# Load processed data
print("Loading processed data...")
with open('processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

skipgram_df = data['skipgram_df']
word2idx = data['word2idx']
idx2word = data['idx2word']
counter = data['counter']
vocab_size = len(word2idx)

print(f"Vocabulary size: {vocab_size}")
print(f"Number of skip-gram pairs: {len(skipgram_df)}")

# Precompute negative sampling distribution below
# Get words counts aligned with vocabulary indices
word_counts = torch.zeros(vocab_size)
for word, idx in word2idx.items():
    if word in counter:
        word_counts[idx] = counter[word]

# Apply 3/4 power smoothing
powered_counts = word_counts ** 0.75

# Normalize to create probability distribution
neg_sampling_probs = powered_counts / powered_counts.sum()

# Device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Move negative sampling probs to device
neg_sampling_probs = neg_sampling_probs.to(device)

# Dataset and DataLoader
dataset = SkipGramDataset(skipgram_df)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# Model, Loss, Optimizer
model = Word2Vec(vocab_size, EMBEDDING_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss(reduction="none")

def make_targets(center, context, vocab_size):
    """
    Generate negative samples for each positive (center, context) pair.
    Ensures that negative samples do not include the positive context word
    or the center word itself.
    
    Args:
        center: [batch_size] tensor of center word indices
        context: [batch_size] tensor of positive context word indices
        vocab_size: size of vocabulary
    
    Returns:
        positive_context: [batch_size] tensor of positive context words
        negative_samples: [batch_size, NEGATIVE_SAMPLES] tensor of negative samples
    """
    batch_size = center.size(0)
    
    # Sample negative examples from the precomputed distribution
    neg = torch.multinomial(
        neg_sampling_probs,
        batch_size * NEGATIVE_SAMPLES,
        replacement=True
    ).view(batch_size, NEGATIVE_SAMPLES)
    
    # Check for collisions: negative samples should not match context
    # context.unsqueeze(1): [batch_size, 1] for broadcasting
    mask = (neg == context.unsqueeze(1))
    
    # Resample only the colliding positions
    while mask.any():
        n_bad = mask.sum().item()  # Count how many collisions occurred
        neg[mask] = torch.multinomial(neg_sampling_probs, n_bad, replacement=True)
        # Recheck for collisions
        mask = (neg == context.unsqueeze(1))
    
    return context, neg

# Training loop
print(f"\nTraining for {EPOCHS} epochs...")
model.train()

for epoch in range(EPOCHS):
    total_loss = 0
    num_batches = 0
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for center_words, context_words in progress_bar:
        # Move to device
        center_words = center_words.to(device)
        context_words = context_words.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Get positive and negative samples
        positive_context, negative_context = make_targets(center_words, context_words, vocab_size)
            
        # POSITIVE loss: [B, 1] -> [B]
        pos_scores = model(center_words, positive_context.unsqueeze(1))
        pos_labels = torch.ones_like(pos_scores)
        pos_loss = criterion(pos_scores, pos_labels).squeeze(1)   # [B]

        # NEGATIVE loss: [B, K] -> sum over K -> [B]
        neg_scores = model(center_words, negative_context)
        neg_labels = torch.zeros_like(neg_scores)
        neg_loss = criterion(neg_scores, neg_labels).mean(dim=1)  # [B]

        # Combine and average over batch
        loss = (pos_loss + neg_loss).mean()

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Track loss
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")

print("\nTraining complete")

# Save embeddings and mappings
embeddings = model.get_embeddings()
with open('word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'word2idx': data['word2idx'], 'idx2word': data['idx2word']}, f)
print("Embeddings saved to word2vec_embeddings.pkl")
