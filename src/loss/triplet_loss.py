import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.loss = nn.TripletMarginLoss(margin=margin)
        
    def forward(self, a_emb, p_emb, n_emb, **batch):
        """
        Computes triplet loss.
        
        Args:
            anchor: Embeddings tensor of anchor [batch_size, embedding_dim]
            positive: Embeddings tensor of positive [batch_size, embedding_dim]
            negative: Embeddings tensor of negative [batch_size, embedding_dim]
            
        Returns:
            torch.Tensor: loss value
        """
        loss = self.loss(a_emb, p_emb, n_emb)
        return {"loss": loss}