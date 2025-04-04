import numpy as np
from tqdm import tqdm
from src.wf_config import CONFIG

class WeightedKNN:
    def __init__(self, k=5, k_reco=5, n_rounds=10):
        self.k = k
        self.k_reco = k_reco
        self.n_rounds = n_rounds
        self.weights = None
        self.X_train = None
        self.y_train = None
    
    def _weighted_distance(self, x1, x2):
        """Calculate weighted Manhattan distance"""
        
        return np.sum(self.weights * np.abs(x1 - x2))
    
    def fit(self, X, y):
        """Train the weighted k-NN classifier"""
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        n_features = self.X_train.shape[1]
        
        # Initialize weights randomly between 0.5 and 1.5
        self.weights = np.random.uniform(0.5, 1.5, n_features)
        
        # Weight adjustment process
        for _ in tqdm(range(self.n_rounds), desc="Training weighted k-NN"):
            for i in range(len(self.X_train)):
                # Calculate distances to all other points
                distances = []
                for j in range(len(self.X_train)):
                    if i == j:
                        continue
                    dist = self._weighted_distance(self.X_train[i], self.X_train[j])
                    distances.append((dist, self.y_train[j]))
                
                # Sort by distance
                distances.sort(key=lambda x: x[0])
                
                # Get k_reco nearest neighbors from same and different classes
                S_good = [d for d in distances if d[1] == self.y_train[i]][:self.k_reco]
                S_bad = [d for d in distances if d[1] != self.y_train[i]][:self.k_reco]
                
                if not S_good or not S_bad:
                    continue
                
                # Weight recommendation step
                d_maxgood = max([d[0] for d in S_good])
                n_bad = []
                for feat in range(n_features):
                    d_maxgood_feat = max([np.abs(self.X_train[i][feat] - self.X_train[j][feat]) 
                                        for d, j in S_good])
                    n_bad_feat = sum([1 for d, j in S_bad 
                                    if np.abs(self.X_train[i][feat] - self.X_train[j][feat]) <= d_maxgood_feat])
                    n_bad.append(n_bad_feat)
                
                # Weight adjustment
                min_n_bad = min(n_bad)
                for feat in range(n_features):
                    if n_bad[feat] != min_n_bad:
                        # Reduce weight
                        delta = self.weights[feat] * 0.01 * (n_bad[feat] / self.k_reco)
                        N_bad = sum([1 for d, _ in S_bad if d <= d_maxgood])
                        delta *= (0.2 + N_bad / self.k_reco)
                        self.weights[feat] -= delta
    
    def predict(self, X):
        """Make predictions using weighted k-NN"""
        X = np.array(X)
        predictions = []
        
        for x in X:
            # Compute distances to all training points
            distances = [(self._weighted_distance(x, self.X_train[i]), self.y_train[i])
                        for i in range(len(self.X_train))]
            
            # Get k nearest neighbors
            distances.sort(key=lambda x: x[0])
            neighbors = distances[:self.k]
            
            # Predict based on unanimous vote (as in paper)
            neighbor_classes = [label for (dist, label) in neighbors]
            if all(c == 1 for c in neighbor_classes):  # All say monitored
                predictions.append(1)
            else:
                predictions.append(0)  # Non-monitored or disagreement
        
        return np.array(predictions)