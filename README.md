# Breast Cancer Prediction using Stacking Ensemble Learning and Image Recognition with OpenAI Assistants API

## Overview
Predict breast cancer using a stacking ensemble of classification algorithms (Random Forest, XGBoost, SVM) to generate meta-features, which is then further trained using Logistic Regression as the meta-model and also image recognition using CNN. These models are integrated with OpenAI Assistants API using custom function calling schema to perform predictions based on natural language.  

## Ensemble Learning

### Dataset
The dataset used in this project comprises features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. These features describe the characteristics of the cell nuclei present in the image. For more details about the dataset, refer to the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29).

### Methodology

The project adopts a stacking ensemble technique. The following steps outline the methodology:

1. **Data Splitting**: The data is split into training, validation, and test sets.
2. **Training Base Models**: Three models—Random Forest, XGBoost, and SVM—are trained on the training set.
3. **Generate Meta-Features**: The trained models make predictions on the validation set, which are used as meta-features.
4. **Train Meta-Model**: A Logistic Regression model is trained on the meta-features.
5. **Making Predictions**: For the test set, predictions are first obtained from each of the base models, then combined, and the meta-model provides the final prediction.

### Evaluation

- **Stacking Ensemble Test Accuracy**: 0.9649
- **Precision**: 0.9756
- **Recall**: 0.9302
- **F1-Score**: 0.9524
- **AUC-ROC**: 0.9581

#### Confusion Matrix:

<img width="482" alt="confusion_matrix" src="https://github.com/cybersamurai2410/BreastCancer_Prediction/assets/66138996/1dd19c04-adbb-4358-abe0-3e125f053fa6">

- 70 True Positives (correctly predicted Malignant)
- 40 True Negatives (correctly predicted Benign)
- 3 False Negatives (incorrectly predicted Benign)
- 1 False Positive (incorrectly predicted Malignant)

## Image Recognition 

### Dataset
### Methodology
### Evaluation

## OpenAI Assistants API
### Running Structured Data Classification
```python
# Calling enemble learning model
classify_cancer("Classify these features: [5.1, 3.5, 1.4, 0.2]", thread_id="thread_abc123")
```
**Console Output:**
```bash
Running enemble learning model...
Final response: "The model predicts a benign tumor based on the provided numerical features. 
Benign tumors are generally non-cancerous and do not spread to other parts of the body. 
However, monitoring for any changes and follow-up screenings are recommended."
```

---

### Running Image Classification
```python
# Calling image classification
classify_cancer("Classify this image: breast_scan.jpg", thread_id="thread_abc123")
```
**Console Output:**
```bash
Running image classification...
Final response: "The model predicts a malignant tumor based on the medical image. 
Malignant tumors exhibit uncontrolled cell growth and may spread to surrounding tissues. 
Further diagnostic tests, such as a biopsy or MRI, are recommended to confirm the diagnosis."
```
