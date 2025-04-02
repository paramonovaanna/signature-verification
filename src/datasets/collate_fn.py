import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    for key in dataset_items[0].keys():
        if key == "labels" or key == "user":
            result_batch[key] = torch.tensor([elem[key] for elem in dataset_items]).long()
        else:
            result_batch[key] = torch.cat(
                [elem[key].unsqueeze(0) for elem in dataset_items], dim=0).float()
    
    return result_batch