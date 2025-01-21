import numpy as np 


def confusion_matrix_values(pred, true):
    tp = np.sum((pred == 1) & (true == 1))
    fp = np.sum((pred == 1) & (true == 0))
    tn = np.sum((pred == 0) & (true == 0))
    fn = np.sum((pred == 0) & (true == 1))
    
    return tp, fp, tn, fn


def f1_score(pred, true):
    tp, fp, tn, fn = confusion_matrix_values(pred, true)
    
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    
    # The final metric you are tested on is the F1 score. 
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1
    

def bin_predictions(msna_peaks, ecg_peaks):
    n = len(ecg_peaks) - 1
    
    pred = np.zeros(n, dtype = bool)
    for i in range(n):
        if np.any((msna_peaks >= ecg_peaks[i]) & (msna_peaks < ecg_peaks[i+1])): 
            pred[i] = True
        else:
            pred[i] = False
        
    return pred
    

def msna_metric(pred_peaks, true_peaks, ecg_peaks):
    """
    The metric we use for this mini-project is the F1 score. To compute the
    F1 score, we bin the signal based on the R-wave peaks of the ECG signal
    and then compute it on the binned values as a classification task. 
    
    Args:
        pred_peaks (np.ndarray): An array of integers representing where each 
            MSNA peak was found.
        true_peaks (np.ndarray): An array of integers representing where each 
            ground truth MSNA peak is. This can be computed as `peaks_from_bool_1d(df["Burst"])`
        true_peaks (np.ndarray): An array of integers representing where each 
            ground truth ECG R-wave peak is. This can be computed as `peaks_from_bool_1d(df["ECG Peaks"])`
    
    Returns:
        float: The F1 score. 
        
    """
    pred = bin_predictions(pred_peaks, ecg_peaks)
    true = bin_predictions(true_peaks, ecg_peaks)
    
    return f1_score(pred, true)


def peaks_from_bool_1d(bool_array): 
    return np.where(bool_array)[0]
    
    
    