"""Lives in its own module (not train_xgboost.py) so that joblib can always
resolve this class on unpickling, regardless of whether training was invoked
as `python src/train_xgboost.py` (where the class would otherwise be bound to
`__main__`) or via `import train_xgboost`.
"""
import numpy as np


class SeedEnsembleClassifier:
    """Averages predict_proba across several independently-trained, identically-
    configured classifiers (differing only in random_state). On a small test
    split, a single XGBoost seed can swing test accuracy by a point or more
    just from training randomness; averaging several seeds reduces that
    variance without touching features or risking overfit to the test set.
    """

    def __init__(self, models: list):
        self.models = models

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
