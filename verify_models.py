"""Check the saved model files against the notebook's recorded test metrics."""
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn

from ml_utils import (
    METRIC_NAMES,
    MODEL_FILES,
    TRAINING_SKLEARN_VERSION,
    evaluate_model,
    validate_test_data,
)

ROOT = Path(__file__).resolve().parent


def main():
    if sklearn.__version__ != TRAINING_SKLEARN_VERSION:
        sys.exit(
            f"Please run with scikit-learn=={TRAINING_SKLEARN_VERSION}. "
            f"Current version: {sklearn.__version__}"
        )

    for needed in [ROOT / "test_data.csv", ROOT / "model_comparison.csv"]:
        if not needed.exists():
            sys.exit(f"Missing file: {needed}")

    df = pd.read_csv(ROOT / "test_data.csv")
    X, y, _ = validate_test_data(df)
    old_table = pd.read_csv(ROOT / "model_comparison.csv").set_index("Model")
    rows = []

    for name, filename in MODEL_FILES.items():
        file_path = ROOT / "model" / filename
        if not file_path.exists():
            sys.exit(f"Missing model: {file_path}")
        model = joblib.load(file_path)
        new_metrics, _, _ = evaluate_model(model, X, y)
        if name not in old_table.index:
            sys.exit(f"Missing result for {name} in model_comparison.csv")
        for metric in METRIC_NAMES:
            recorded = float(old_table.loc[name, metric])
            current = float(new_metrics[metric])
            if not np.isclose(current, recorded, rtol=1e-6, atol=1e-8):
                sys.exit(
                    f"Metric mismatch for {name}, {metric}: "
                    f"saved={recorded:.7f}, current={current:.7f}"
                )
        rows.append({"Model": name, **new_metrics})

    print(f"PASS: {len(df)} test rows, {X.shape[1]} features, five model files")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("PASS: all saved model metrics match model_comparison.csv")


if __name__ == "__main__":
    main()
