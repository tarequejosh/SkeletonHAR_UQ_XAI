import numpy as np

class AdaptivePredictionSets:
    """
    Adaptive Prediction Sets (APS) for Split Conformal Prediction.
    Guarantees finite-sample marginal coverage: P(Y \in C(X)) \ge 1 - \alpha.
    """
    def __init__(self, alpha=0.10, randomized=False, seed=42):
        self.alpha = alpha
        self.randomized = randomized
        self.seed = seed
        self.q_hat = None
        self.rng = np.random.default_rng(seed)

    def calibrate(self, cal_probs, cal_labels):
        """
        cal_probs: (N_cal, num_classes) predicted probability distribution on calibration set
        cal_labels: (N_cal,) true integer class labels
        """
        N = len(cal_labels)
        scores = []
        
        for i in range(N):
            p = cal_probs[i]
            y = cal_labels[i]
            
            # Sort probabilities descending
            sorted_indices = np.argsort(-p)
            sorted_probs = p[sorted_indices]
            
            # Find rank of true label
            rank = np.where(sorted_indices == y)[0][0]
            
            # Cumulative probability up to true class
            cum_prob = np.sum(sorted_probs[:rank + 1])
            
            if self.randomized:
                u = self.rng.uniform(0, 1)
                score = cum_prob - u * sorted_probs[rank]
            else:
                score = cum_prob
                
            scores.append(score)
            
        scores = np.array(scores)
        
        # Conformal quantile: ceil((N + 1) * (1 - alpha)) / N
        # Exactly following Angelopoulos & Bates (2021) "A Gentle Introduction to Conformal Prediction"
        # k = ceil((N + 1) * (1 - alpha)), index in 0-indexed sorted scores is k - 1
        k = int(np.ceil((N + 1) * (1.0 - self.alpha)))
        k = min(max(k, 1), N) # guard against edge cases
        self.q_hat = float(np.sort(scores)[k - 1])
        print(f"APS Calibrated: n={N}, alpha={self.alpha:.2f} (Target Coverage={(1-self.alpha)*100:.1f}%) | k={k}/{N} -> q_hat={self.q_hat:.4f}")
        return self.q_hat

    def predict_sets(self, test_probs):
        """
        Constructs prediction set C(x) for each test sample.
        test_probs: (N_test, num_classes)
        returns: list of lists containing class indices in prediction set
        """
        if self.q_hat is None:
            raise ValueError("Conformal predictor must be calibrated before predict_sets() is called.")
            
        N = len(test_probs)
        prediction_sets = []
        
        for i in range(N):
            p = test_probs[i]
            sorted_indices = np.argsort(-p)
            sorted_probs = p[sorted_indices]
            
            cum_probs = np.cumsum(sorted_probs)
            
            if self.randomized:
                u = self.rng.uniform(0, 1)
                cutoffs = cum_probs - u * sorted_probs
                idx = np.where(cutoffs >= self.q_hat)[0]
            else:
                idx = np.where(cum_probs >= self.q_hat)[0]
                
            if len(idx) > 0:
                k_star = idx[0] + 1
            else:
                k_star = len(p)
                
            pred_set = sorted_indices[:k_star].tolist()
            prediction_sets.append(pred_set)
            
        return prediction_sets

    def evaluate_coverage(self, test_probs, test_labels):
        """
        Evaluates empirical coverage and average set size on test set,
        including breakdown for correct and misclassified predictions.
        """
        prediction_sets = self.predict_sets(test_probs)
        N = len(test_labels)
        
        covered = [test_labels[i] in prediction_sets[i] for i in range(N)]
        set_sizes = np.array([len(s) for s in prediction_sets])
        
        # Point predictions
        point_preds = np.argmax(test_probs, axis=1)
        is_correct = (point_preds == test_labels)
        
        empirical_coverage = np.mean(covered)
        avg_set_size = float(np.mean(set_sizes))
        singleton_prop = float(np.mean(set_sizes == 1))
        
        avg_set_size_correct = float(np.mean(set_sizes[is_correct])) if np.sum(is_correct) > 0 else 0.0
        avg_set_size_misclass = float(np.mean(set_sizes[~is_correct])) if np.sum(~is_correct) > 0 else 0.0
        
        return {
            "target_coverage": float(1.0 - self.alpha),
            "empirical_coverage": float(empirical_coverage),
            "avg_set_size": avg_set_size,
            "avg_set_size_correct": avg_set_size_correct,
            "avg_set_size_misclass": avg_set_size_misclass,
            "singleton_proportion": singleton_prop,
            "set_sizes": set_sizes.tolist(),
            "prediction_sets": prediction_sets
        }
