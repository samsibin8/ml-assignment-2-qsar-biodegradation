"""QSAR Biodegradation: evaluation of five previously trained classifiers."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import streamlit as st
from sklearn.metrics import classification_report

from ml_utils import (
    CLASS_ORDER,
    METRIC_NAMES,
    MODEL_FILES,
    TRAINING_SKLEARN_VERSION,
    DataValidationError,
    evaluate_model,
    validate_test_data,
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "model"
EXAMPLE_CSV = PROJECT_DIR / "test_data.csv"

st.set_page_config(page_title="QSAR Biodegradation | ML Assignment 2", layout="wide")


@st.cache_resource(show_spinner="Loading saved classifier...")
def load_saved_model(filename: str):
    """Load the model once. This application NEVER calls fit()."""
    return joblib.load(MODEL_DIR / filename)


def show_confusion_matrix(matrix, name):
    """Show actual classes on rows and predicted classes on columns."""
    fig, ax = plt.subplots(figsize=(5.5, 4.4))
    heatmap = ax.imshow(matrix, cmap="Blues", vmin=0)
    ax.set_xticks(range(2), CLASS_ORDER)
    ax.set_yticks(range(2), CLASS_ORDER)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title(name + " - Confusion matrix")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
    fig.colorbar(heatmap, ax=ax, shrink=0.78)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


st.title("QSAR Biodegradation Classification")
st.caption("BITS WILP Machine Learning Assignment 2 | UCI QSAR Biodegradation")
st.write(
    "Evaluate saved classification models on a labeled CSV. "
    "The input must contain 41 numeric descriptors and a target (NRB or RB). "
    "The app uses trained models; it does not retrain them."
)

if sklearn.__version__ != TRAINING_SKLEARN_VERSION:
    st.error(
        "The saved models were trained using scikit-learn "
        f"{TRAINING_SKLEARN_VERSION}, but this app is using {sklearn.__version__}. "
        "Use the supplied requirements.txt and deploy with Python 3.12."
    )
    st.stop()

left, right = st.columns([2, 1])
with left:
    upload = st.file_uploader("Upload a labeled test CSV", type=["csv"])
with right:
    model_name = st.selectbox("Inspect model", options=list(MODEL_FILES))

if EXAMPLE_CSV.exists():
    st.download_button(
        "Download the notebook test CSV",
        data=EXAMPLE_CSV.read_bytes(),
        file_name="test_data.csv",
        mime="text/csv",
    )

if upload is None:
    st.info("Upload the test_data.csv to see the comparison and confusion matrix.")
    st.stop()

try:
    uploaded_df = pd.read_csv(upload)
    X, y, extra_columns = validate_test_data(uploaded_df)
except (DataValidationError, pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
    st.error(f"Cannot use this CSV: {exc}")
    st.stop()

st.success(f"Loaded {len(uploaded_df)} rows with {X.shape[1]} input features.")
if extra_columns:
    st.warning("Extra columns are ignored: " + ", ".join(extra_columns))

with st.expander("View uploaded data", expanded=False):
    st.dataframe(uploaded_df.head(10), width="stretch", hide_index=True)
    st.write("Actual class counts:")
    st.write(y.value_counts().reindex(CLASS_ORDER, fill_value=0))

# All models evaluate the SAME CSV; model fitting is intentionally absent.
results = []
predictions_by_name = {}
matrices_by_name = {}
for name, filename in MODEL_FILES.items():
    try:
        saved_model = load_saved_model(filename)
        metrics, predictions, matrix = evaluate_model(saved_model, X, y)
    except Exception as exc:
        st.error(f"Could not evaluate {name} ({filename}): {exc}")
        st.stop()
    results.append({"Model": name, **metrics})
    predictions_by_name[name] = predictions
    matrices_by_name[name] = matrix

comparison_df = pd.DataFrame(results)
st.subheader("All five models - comparison")
st.dataframe(
    comparison_df.style.format({metric: "{:.4f}" for metric in METRIC_NAMES}, na_rep="N/A"),
    width="stretch",
    hide_index=True,
)

if y.nunique() != 2:
    st.warning("AUC is not defined because the uploaded test data contains only one class.")
else:
    winner_f1 = comparison_df.loc[comparison_df["F1"].idxmax(), "Model"]
    winner_auc = comparison_df.loc[comparison_df["AUC"].idxmax(), "Model"]
    st.write(f"Highest F1 on this CSV: **{winner_f1}** | Highest AUC: **{winner_auc}**")

selected_metrics = comparison_df.set_index("Model").loc[model_name]
st.subheader(f"Selected model: {model_name}")
first, second, third = st.columns(3)
for area, names in zip([first, second, third], [METRIC_NAMES[:2], METRIC_NAMES[2:4], METRIC_NAMES[4:]]):
    with area:
        for metric in names:
            value = selected_metrics[metric]
            st.metric(metric, "N/A" if pd.isna(value) else f"{value:.4f}")

one, two = st.columns([1, 1])
with one:
    show_confusion_matrix(matrices_by_name[model_name], model_name)
with two:
    st.markdown("**Classification report**")
    report = classification_report(
        y,
        predictions_by_name[model_name],
        labels=CLASS_ORDER,
        zero_division=0,
    )
    st.code(report, language="text")
    st.caption("RB is treated as the positive class for precision, recall and F1.")

st.caption("Predictions and metrics use only the uploaded CSV and saved model files in model/.")
