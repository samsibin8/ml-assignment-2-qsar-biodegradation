# QSAR Biodegradation Classification

Machine Learning Assignment 2 | BITS WILP M.Tech (AIML)

**Status:** Five classification models trained and evaluated; Streamlit application deployed.

## a. Problem statement

The objective is to predict whether a chemical is ready biodegradable (RB) or not ready biodegradable (NRB) using 41 numerical molecular descriptors. Five machine learning classifiers are trained on the same dataset, evaluated on a held-out test set, and compared using six classification metrics. A Streamlit application loads the trained models and evaluates an uploaded, labeled test CSV without retraining.

## b. Dataset description

- **Dataset:** QSAR Biodegradation, UCI Machine Learning Repository.
- **Dataset size:** 1,055 chemical records and 41 numerical input features.
- **Target:** Ready biodegradable (RB, positive class) and not ready biodegradable (NRB).
- **Class distribution:** 356 RB and 699 NRB records.
- **Missing values:** None in the dataset used.
- **Train/test split:** 844 training records and 211 test records. The same test partition was used to evaluate all five classifiers.
- **Original dataset:** https://archive.ics.uci.edu/dataset/254/qsar%2Bbiodegradation
- **Dataset citation:** Mansouri, K., Ringsted, T., Ballabio, D., Todeschini, R., & Consonni, V. (2013). *QSAR biodegradation*. UCI Machine Learning Repository. https://doi.org/10.24432/C5H60M
- **License:** CC BY 4.0.

## c. GitHub repository link

https://github.com/samsibin8/ml-assignment-2-qsar-biodegradation

## d. Models used and evaluation results

The classifiers are Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Gaussian Naive Bayes, and Support Vector Machine (SVM). StandardScaler is included in the saved pipelines for Logistic Regression, KNN and SVM. All models use the same held-out dataset, and RB is treated as the positive class for Precision, Recall and F1.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8626 | 0.9136 | 0.8000 | 0.7887 | 0.7943 | 0.6912 |
| Decision Tree | 0.8152 | 0.8656 | 0.7162 | 0.7465 | 0.7310 | 0.5906 |
| KNN | 0.8389 | 0.8971 | 0.7403 | 0.8028 | 0.7703 | 0.6478 |
| Naive Bayes | 0.6967 | 0.8650 | 0.5276 | 0.9437 | 0.6768 | 0.4972 |
| Support Vector Machine | 0.8483 | 0.9189 | 0.7910 | 0.7465 | 0.7681 | 0.6562 |

### Observations about model performance

- **Logistic Regression:** TP=56, TN=126, FP=14, FN=15. This model achieved the highest F1, accuracy and MCC. Its precision and recall are reasonably balanced for RB.
- **Decision Tree:** TP=53, TN=119, FP=21, FN=18. Lower F1 and accuracy than Logistic Regression; it made more false-positive RB predictions.
- **KNN:** TP=57, TN=120, FP=20, FN=14. Good RB recall, but more false positives than Logistic Regression.
- **Naive Bayes:** TP=67, TN=80, FP=60, FN=4. Highest recall, but low precision and many false positives. The trade-off reduced its overall F1 and accuracy.
- **Support Vector Machine:** TP=53, TN=126, FP=14, FN=18. Highest AUC, showing strong ranking ability on this held-out split, but lower recall and F1 than Logistic Regression.

### Overall winner

**Logistic Regression** is the overall winner using F1 on the held-out test set (F1 = 0.7943). It also recorded the highest accuracy (0.8626) and MCC (0.6912). SVM recorded the highest AUC (0.9189); Naive Bayes recorded the highest recall (0.9437) but made many false-positive RB predictions. These conclusions apply to this test split, not necessarily to other datasets.

## e. Streamlit web application

**Live application:** https://qsarbiodegradation.streamlit.app/

The application accepts a labeled test CSV containing `feature_01` through `feature_41` and `target`. It provides a dropdown for the five saved models, a comparison table with all six metrics, the selected model metrics, a confusion matrix, and a classification report. It loads `.joblib` model files from `model/` and does not retrain any model on uploaded test data.

## f. Reproducibility and project files

- `01_qsar_baseline.ipynb`: baseline data checks and Logistic Regression evaluation.
- `02_qsar_all_models.ipynb`: five-model training, evaluation and saved artifacts.
- `biodeg.csv`: original UCI dataset.
- `test_data.csv`: held-out labeled test data (211 records).
- `model_comparison.csv`: six evaluation metrics for the five classifiers.
- `model_observations.csv`: model-wise observations and confusion-matrix counts.
- `model/`: five saved trained model files.
- `app.py`, `ml_utils.py`: Streamlit application and evaluation helpers.
- `requirements.txt`: deployment dependencies.
- `Screenshots/02_five_models_comparison.png`: execution evidence from BITS Virtual Lab.

For local evaluation, install the dependencies with `pip install -r requirements.txt`, run `python verify_models.py` to check the saved model metrics, and launch the app with `streamlit run app.py`.

## g. BITS Virtual Lab evidence

The models were executed in BITS Virtual Lab. The screenshot under `Screenshots/02_five_models_comparison.png` shows the executed notebook and observations. The PDF submission includes this screenshot with its session URL parameters concealed for security.
