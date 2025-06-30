from torch import nn
from models.GRU import MultiLayerGRU


class POSTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, tagset_size, num_layers, dropout, token2idx):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=token2idx['<PAD>'])
        self.gru = MultiLayerGRU(embedding_dim, hidden_dim, num_layers, dropout)
        self.fc = nn.Linear(hidden_dim, tagset_size)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (batch, seq_len, embedding_dim)
        out, _ = self.gru(emb)   # (batch, seq_len, hidden_dim)
        logits = self.fc(out)    # (batch, seq_len, tagset_size)
        return logits
