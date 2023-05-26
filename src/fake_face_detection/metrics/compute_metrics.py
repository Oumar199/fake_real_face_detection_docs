
import numpy as np
import evaluate

metrics = {
    'f1': evaluate.load('f1'),
    'accuracy': evaluate.load('accuracy'),
    'roc_auc': evaluate.load('roc_auc', 'multiclass')
}

def compute_metrics(p): # some part was got from https://huggingface.co/blog/fine-tune-vit
    
    predictions, label_ids = p
    
    metric = metrics['accuracy'].compute(predictions = np.argmax(predictions, axis = 1), references=label_ids)
    
    f1_score = metrics['f1'].compute(predictions=np.argmax(predictions, axis = 1), references=label_ids)
    
    metric.update(f1_score)
    
    try:
        
        auc = metrics['roc_auc'].compute(prediction_scores=predictions, references=label_ids)
    
        metric.update(auc)
        
    except:
        
        pass
        
    return metric
    
    
