
def fake_face_collator(batch):
    """The data collator for training vision transformer models on fake and real face dataset

    Args:
        batch (dict): A dictionary containing the pixel values and the labels

    Returns:
        dict: The final dictionary
    """
    batch['pixel_values'] = batch['pixel_values'][0]
    
    return batch
