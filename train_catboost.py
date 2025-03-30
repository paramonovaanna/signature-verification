import torch

import os

from src.utils.init_utils import set_random_seed

import hydra 
from hydra.utils import instantiate

from src.siamese_data.data_utils import get_dataloaders, get_inference_dataloaders

from src.embeddings import EmbeddingsExtractor

from src.models.siamese_networks import SiameseNetwork

from src.utils.io_utils import ROOT_PATH

import numpy as np

from sklearn.metrics import accuracy_score, roc_curve, auc

from catboost import CatBoostClassifier

@hydra.main(version_base=None, config_path="src/configs", config_name="catboost")
def main(config):
    set_random_seed(config.seed)

    if config.device.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device.device

    model1 = instantiate(config.model._model_).to(device)
    model1 = SiameseNetwork(model1).to(device)
    #model2 = instantiate(config.model2._model_).to(device)

    dataloaders, batch_transforms = get_dataloaders(config, device)
    test_dataloaders, batch_transforms = get_inference_dataloaders(config, device)
    dataloaders.update(test_dataloaders)

    extractor = EmbeddingsExtractor(
        models=[model1, None], 
        pretrained_paths=config.embeddings.pretrained_paths,
        device=device,
        device_tensors=config.device.device_tensors, 
        siamese=True
    )
    emb, labels = extractor.extract(
        save_dir=config.embeddings.save_dir,
        filename=config.embeddings.filename,
        dataloaders=dataloaders,
    )

    classifier = instantiate(config.catboost)
    print("Training...")
    classifier.fit(
        emb["train"], labels["train"],
        eval_set=(emb["inference"], labels["inference"])
    )
    classifier.save_model('catboost_model.cbm')

    loaded_model = CatBoostClassifier()
    loaded_model.load_model('catboost_model.cbm')

    val_pred = loaded_model.predict(emb["inference"])
    val_acc = np.mean(val_pred == labels["inference"])
    print(f"Inference original accuracy: {val_acc:.4f}")

    y_pred_proba = loaded_model.predict_proba(emb["inference"])[:, 1]
    fpr, tpr, thresholds = roc_curve(labels["inference"], y_pred_proba)
    optimal_idx = np.argmax(abs(tpr - fpr))
    eer = np.mean((fpr[optimal_idx], tpr[optimal_idx]))
    print("EER:", eer)
    optimal_threshold = thresholds[optimal_idx]
    y_pred = (y_pred_proba > optimal_threshold).astype(int)
    acc = accuracy_score(labels["inference"], y_pred)
    print(f"Threshold {optimal_threshold:.2f}: eer_accuracy {acc:.4f}")


    y_pred = (y_pred_proba > 0.5).astype(int)
    acc = accuracy_score(labels["inference"], y_pred)
    print(f"Prob accuracy on inference set: {acc:.4f}")


    evals_result = classifier.get_evals_result()
    best_accuracy = evals_result['validation']['Accuracy'][classifier.best_iteration_]
    print(f"Internal best accuracy: {best_accuracy:.4f}")

    # Получаем вероятности
    """y_pred_proba = classifier.predict_proba(emb["inference"], ntree_end=classifier.best_iteration_)[:, 1]


    print("Evaluating...")
    val_pred = classifier.predict(emb["test"])
    print(val_pred.shape, emb["test"].shape, labels["test"].shape)
    val_acc = np.mean(val_pred == labels["test"])
    print(f"Test accuracy: {val_acc:.4f}")
    
    test_pred = classifier.predict(emb["inference"])
    test_acc = np.mean(test_pred == labels["inference"])
    print(f"Inference accuracy: {test_acc:.4f}")"""
    
    """save_dir = ROOT_PATH / "checkpoints" / "catboost"
    os.makedirs(save_dir, exist_ok=True)
    classifier.model.save_model(os.path.join(save_dir, "coatnetL-0.9split_classifier.cbm"))
    print(f"Model saved to {save_dir}/coatnetL-0.9split_classifier.cbm")"""


if __name__ == "__main__":
    main()