import numpy as np
from sklearn.model_selection import train_test_split
from src.data_loader import load_dataset
from src.feature_extractor import extract_all_features
from src.knn_attack import WeightedKNN
from src.evaluator import evaluate_results
from src.wf_config import CONFIG

def main():
    # 1. Load and preprocess data
    print("Loading dataset...")
    sequences, y, site_ids = load_dataset()
    
    # 2. Extract features
    print("\nExtracting features...")
    X = []
    for seq in sequences:
        features = extract_all_features(seq)
        X.append(features)
    
    # 3. Split into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["TEST_SIZE"], 
        random_state=CONFIG["RANDOM_SEED"], stratify=y)
    
    # 4. Train weighted k-NN
    print("\nTraining classifier...")
    wknn = WeightedKNN(
        k=CONFIG["K_NEIGHBORS"],
        k_reco=CONFIG["K_RECO"],
        n_rounds=CONFIG["N_ROUNDS"])
    wknn.fit(X_train, y_train)
    
    # 5. Evaluate
    print("\nEvaluating...")
    y_pred = wknn.predict(X_test)
    metrics = evaluate_results(y_test, y_pred)
    
    # 6. Save results
    print("\nTraining complete!")
    print(f"Final TPR: {metrics['tpr']:.4f}, FPR: {metrics['fpr']:.4f}")

if __name__ == "__main__":
    main()