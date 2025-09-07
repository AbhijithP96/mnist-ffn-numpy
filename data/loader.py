import numpy as np

class Loader:

    def get_batches(self, X: np.ndarray, y: np.ndarray, batch_size: int) -> list:
        """Create batches of data.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target labels.
            batch_size (int): Size of each batch.

        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: A list of batches, each containing input features and target labels.
        """
        num_samples = X.shape[0]
        batches = []
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batches.append((X[start:end], y[start:end]))
        return batches