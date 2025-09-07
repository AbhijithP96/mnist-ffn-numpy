import numpy as np

class CrossEntropyLoss:

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return self._cross_entropy_loss(y_true, y_pred)

    def _cross_entropy_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the cross-entropy loss between true labels and predictions.
        
        Args:
            y_true (np.ndarray): True labels (one-hot encoded).
            y_pred (np.ndarray): Predicted probabilities.
            
        Returns:
            float: Cross-entropy loss.
        """
        m = y_true.shape[1]
        # Clip predictions to avoid log(0)
        y_pred_clipped = np.clip(y_pred, 1e-15, 1 - 1e-15)
        loss = -np.sum(y_true * np.log(y_pred_clipped)) / m
        return loss