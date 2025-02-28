from transformers import ConvNextImageProcessor


def convnext_transform():
    processor = ConvNextImageProcessor(
        return_tensors="pt"
    )

    def transform(img):
        return processor.preprocess(img, return_tensors="pt")["pixel_values"]
    
    return transform


