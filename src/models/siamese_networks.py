import torch
import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

class SiameseNetwork(nn.Module):
    def __init__(self, model, embedding_dim=512):
        super().__init__()
        self.backbone = model

        num_features = self.backbone.model.head.fc.in_features
        self.backbone.model.head.fc = nn.Sequential(
            nn.Linear(num_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward_one(self, x):
        return self.backbone(x)["logits"]
    
    def forward(self, anchor, positive, negative, **batch):
        anchor_embedding = self.forward_one(anchor)
        positive_embedding = self.forward_one(positive)
        negative_embedding = self.forward_one(negative)

        return {"a_emb": anchor_embedding, "p_emb": positive_embedding, "n_emb": negative_embedding}

    def test_forward(self, reference, sig, **batch):
        reference_embedding = self.forward_one(reference)
        sig_embedding = self.forward_one(sig)

        return {"r_emb": reference_embedding, "s_emb": sig_embedding}
