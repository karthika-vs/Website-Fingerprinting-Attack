from sklearn.metrics import classification_report, confusion_matrix

def evaluate_results(y_true, y_pred):
    """Calculate and print evaluation metrics"""
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Non-monitored', 'Monitored']))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    # Calculate True Positive Rate and False Positive Rate
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    
    print(f"\nTrue Positive Rate: {tpr:.4f}")
    print(f"False Positive Rate: {fpr:.4f}")
    
    return {
        'tpr': tpr,
        'fpr': fpr,
        'precision': tp / (tp + fp),
        'recall': tpr,
        'accuracy': (tp + tn) / (tp + tn + fp + fn)
    }