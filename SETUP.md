# Working with the QSAR baseline notebook

1. Open `01_qsar_baseline.ipynb` in BITS Virtual Lab.
2. Install required packages in the notebook environment if needed, e.g. `%pip install -r requirements.txt`.
3. Run the notebook. If the UCI download is blocked, download the original `biodeg.csv` from the [UCI QSAR Biodegradation page](https://archive.ics.uci.edu/dataset/254/qsar+biodegradation), extract it and place it beside the notebook.
4. Confirm that the dataset has 1,055 rows, 41 feature columns, and a target column. Review the real metrics and observations produced by the notebook.
5. Save the executed notebook and commit the generated `test_data.csv`, `baseline_metrics.csv`, and `model/logistic_regression.joblib` along with the source files. Capture one BITS Virtual Lab screenshot.

This is the first model baseline only. The remaining four classifiers and the Streamlit app are planned but not yet implemented.
