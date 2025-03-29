import torch


def train_collate_fn(dataset_items):
    result_batch = {}
    result_batch["anchor"] = torch.cat(
        [elem["anchor"].unsqueeze(0) for elem in dataset_items], dim=0).float()
    result_batch["positive"] = torch.cat(
        [elem["positive"].unsqueeze(0) for elem in dataset_items], dim=0).float()
    result_batch["negative"] = torch.cat(
        [elem["negative"].unsqueeze(0) for elem in dataset_items], dim=0).float()
    return result_batch

def test_collate_fn(dataset_items):
    result_batch = {}
    result_batch["reference"] = torch.cat(
        [elem["reference"].unsqueeze(0) for elem in dataset_items], dim=0).float()
    result_batch["sig"] = torch.cat(
        [elem["sig"].unsqueeze(0) for elem in dataset_items], dim=0).float()
    result_batch["labels"] = torch.tensor(
        [elem["labels"] for elem in dataset_items]).long()
    
    return result_batch