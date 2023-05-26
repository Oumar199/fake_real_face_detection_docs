
from transformers import Trainer
import torch

def get_custom_trainer(weights: torch.Tensor): 
    
    class CustomTrainer(Trainer): # got from https://huggingface.co/docs/transformers/main_classes/trainer
        
        def compute_loss(self, model, inputs, return_outputs=False):
            
            # recuperate labels
            labels = inputs.get("labels")
            
            # forward pass
            outputs = model(**inputs)

            # recuperate logits
            logits = outputs.get("logits")
            
            # compute custom loss (passing the weights)
            loss_fct = nn.CrossEntropyLoss(weight=weights)
            
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
            return (loss, outputs) if return_outputs else loss
    
    return CustomTrainer
