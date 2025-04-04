from catboost import CatBoostClassifier
from catboost import Pool

import pandas as pd

from hydra.utils import instantiate

class CBClassifier:

    def __init__(self, data, metrics, from_pretrained=None, save_filename=None, **config):
        self.classifier = CatBoostClassifier(**config)

        self.labels = {partition: data[partition]["labels"] for partition in data.keys()}
        self.pools = self._initialise_pools(data, self.labels)

        self.save_filename = save_filename

        self.metrics = metrics

        self.from_pretrained = from_pretrained
        if self.from_pretrained:
            self._init_from_pretrained(from_pretrained)

    def classify(self):
        if self.from_pretrained is None:
            self.train()

        logs = {}
        for part in self.metrics.keys():
            probs = self.predict(self.pools[part])
            logs[part] = {}
            for met in self.metrics[part]:
                logs[part].update({met.name: met(probs, self.labels[part])})
        return logs

    def train(self):
        self.classifier.fit(self.pools["train"], eval_set=self.pools.get("validation", None))
        if self.save_filename:
            self._save_weights(self.save_filename)

    def predict(self, pool):
        probs = self.classifier.predict_proba(pool)
        return probs

    def _initialise_pools(self, data, labels):
        feature_pools = {}
        for partition in data.keys():
            features = {key: list(value) for key, value in data[partition].items() if key != "labels" and key != "user"}
            df = pd.DataFrame(features)
            feature_pools[partition] = Pool(
                data=df,
                label=labels[partition],
                embedding_features=list(features.keys())
            )
        return feature_pools

    #features = self._separate_sig_features(data[partition])

            #features_amount = len([feature for feature in data[partition].keys() if feature != "labels" and feature != "user"])
            #embedding_features = list(range(features_amount))
            #print(features[0])

    def _separate_sig_features(self, data):
        all_features = []
        sig_amount = len(data["user"])
        for i in range(sig_amount):
            instance_features = [list(data[feature][i]) for feature in data.keys() if feature != "labels" and feature != "user"]
            all_features.append(instance_features)
        return all_features

    def _save_weights(self, filename):
        save_dir = ROOT_PATH / "saved" / "catboost"
        os.makedirs(save_dir, exist_ok=True)
        path = save_dir / filename
        self.classifier.save_model(path)

    def _init_from_pretrained(self, path):
        self.classifier.load_model(path)


