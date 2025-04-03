import torch
import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights

class SiameseNetwork(nn.Module):
    def __init__(self, model, embedding_dim=512):
        super().__init__()
        self.backbone = model
        self.name = f"siam_{self.backbone.name}"

        num_features = self.backbone.model.head.fc.in_features
        self.backbone.model.head.fc = nn.Sequential(
            nn.Linear(num_features, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward_one(self, x):
        image = {"img": x}
        return self.backbone(**image)["logits"]

    def forward_triplet(self, anchor, pos, neg, **batch):
        anchor_embedding = self.forward_one(anchor)
        positive_embedding = self.forward_one(pos)
        negative_embedding = self.forward_one(neg)
        return {"a_emb": anchor_embedding, "p_emb": positive_embedding, "n_emb": negative_embedding}

    def forward_pair(self, ref, sig, **batch):
        reference_embedding = self.forward_one(ref)
        sig_embedding = self.forward_one(sig)

        return {"r_emb": reference_embedding, "s_emb": sig_embedding}

    def forward(self, **batch):
        if batch.get("anchor", None) is not None:
            return self.forward_triplet(**batch)
        elif batch.get("ref", None) is not None:
            return self.forward_pair(**batch)

    def features(self, **batch):
        return self.forward(**batch)