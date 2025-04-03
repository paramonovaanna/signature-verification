import numpy as np

from abc import abstractmethod

class DistanceClassifier:
    def __init__(self, user2emb,
                m_values,
                num_users,
                labels,
                embeddings, **kwargs):
        """
        Initialize DistanceClassifier.
        Args:
            dataset: PreprocessedDataset instance containing signature data
        """
        self.num_users = num_users
        self.m_values = np.arange(m_values[0], m_values[1])

        self.user_embeddings = self._init_user_embeddings(embeddings, user2emb, labels)

    @abstractmethod
    def _init_user_embeddings(self):
        raise NotImplemented()

    @abstractmethod
    def calculate_distances(self):
        raise NotImplemented()

    @abstractmethod
    def classify(self):
        raise NotImplemented()