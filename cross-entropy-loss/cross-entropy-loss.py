import numpy as np

def cross_entropy_loss(y_true, y_pred):
    """
    Compute average cross-entropy loss for multi-class classification.
    """
    # Get the number of samples (N)
    y_pred = np.array(y_pred)
    N = len(y_true)
    
    # 1. Extract the predicted probabilities for the correct classes
    correct_probs = y_pred[np.arange(N), y_true]
    
    # 2. Compute the negative average of the log probabilities
    loss = -np.mean(np.log(correct_probs))
    
    return loss