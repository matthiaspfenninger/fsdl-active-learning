import numpy as np
import torch

def get_random_samples(pool_size,sample_size):
    
    #return sample size of random indices 
    
    return np.random.randint(pool_size, size=sample_size)

def get_least_confidence_samples(predictions, sample_size):

    conf = []
    indices = []
    for idx,prediction in enumerate(predictions):
        most_confident = np.max(prediction)
        conf.append(most_confident)
        indices.append(idx)
            
    conf = np.asarray(conf)
    indices = np.asarray(indices)
        
    return indices[np.argsort(conf)][:sample_size]

def get_top2_confidence_margin_samples(predictions, sample_size):

    
    margins = []
    indices = []
        
    for idx,predxn in enumerate(predictions):
        predxn[::-1].sort()
        margins.append(predxn[0]-predxn[1])
        indices.append(idx)
    margins=np.asarray(margins)
    indices=np.asarray(indices)
    least_margin_indices=indices[np.argsort(margins)][:sample_size]
  
    return least_margin_indices