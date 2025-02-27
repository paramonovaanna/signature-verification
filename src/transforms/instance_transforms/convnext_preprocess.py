from transformers import ConvNextImageProcessor


def convnext_preprocessing(cfg):
    processor = ConvNextImageProcessor(
        do_resize=cfg.do_resize,
        size={"shortest_edge": cfg.size.shortest_edge},
        do_normalize=cfg.do_normalize,
        image_mean=cfg.image_mean,
        image_std=cfg.image_std,
        return_tensors="pt"
    )

    def transform(img):
        return processor.preprocess(img, return_tensors="pt")["pixel_values"]
    
    return transform


