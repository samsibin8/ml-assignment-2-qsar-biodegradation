# QSAR Biodegradation Classification

Machine Learning Assignment 2 | BITS WILP M.Tech (AIML)

**Status:** Initial project planning. Dataset validation, model training, evaluation, and deployment are pending.

## a. Problem statement

Compare five classification algorithms for predicting whether a chemical is ready biodegradable from its molecular descriptors. The project will include model evaluation and a Streamlit application for evaluating the saved models on uploaded test data.

## b. Dataset description

The selected dataset is **QSAR Biodegradation**, from the UCI Machine Learning Repository.

| Property | Description |
|---|---|
| Records | 1,055 |
| Input features | 41 numerical molecular descriptors |
| Target | Ready biodegradable (RB) or not ready biodegradable (NRB) |
| Class counts | 356 RB and 699 NRB |
| Planned target encoding | RB = 1; NRB = 0 |
| Missing values | UCI reports none; this will be verified during exploration |

These figures are from the source documentation. The next step is to load the original file in BITS Virtual Lab and check its shape, labels, missing values, and duplicate records.

- Dataset: https://archive.ics.uci.edu/dataset/254/qsar%2Bbiodegradation
- Citation: Mansouri, K., Ringsted, T., Ballabio, D., Todeschini, R., and Consonni, V. (2013). *QSAR biodegradation*. UCI Machine Learning Repository. https://doi.org/10.24432/C5H60M
- Dataset license: Creative Commons Attribution 4.0.

## c. GitHub repository link

https://github.com/samsibin8/ml-assignment-2-qsar-biodegradation.git

## d. Models used - implementation plan

The following classifiers will be implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors Classifier
4. Gaussian Naive Bayes
5. Support Vector Machine

All five models will use the same held-out test split. Any preprocessing that learns from data will be fitted using training data only.

Each model will be evaluated using **Accuracy, ROC AUC, Precision, Recall, F1, and Matthews Correlation Coefficient (MCC)**. RB will be the positive class for binary precision, recall, and F1.

The comparison table, observations for each model, and overall winner will be added after the experiments have been run and reviewed.

## Implementation checklist

- [x] Review the assignment requirements.
- [x] Select a dataset that meets the instance and feature requirements.
- [x] Document the initial project plan.
- [x] Load and inspect the dataset in BITS Virtual Lab.
- [x] Check duplicates and create the training and test partitions.
- [x] Implement and evaluate all five classifiers.
- [x] Save trained model pipelines in `model/`.
- [x] Export `test_data.csv` with all input features and the target.
- [ ] Build the Streamlit app with CSV upload, model selection, metrics, and a confusion matrix or classification report.
- [ ] Evaluate uploaded test data using saved models without retraining.
- [ ] Add dependencies to `requirements.txt` and deploy on Streamlit Community Cloud.
- [ ] Add the actual results, model observations, and working links to this README.
- [x] Capture a screenshot of actual execution in BITS Virtual Lab.
- [ ] Prepare the submission PDF with the repository link, app link, lab screenshot, and README content in the required order.
