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
5. **Inference**: For the test set, predictions are first obtained from each of the base models, then combined, and the meta-model provides the final prediction.

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
The [CBIS-DDSM](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset) (Curated Breast Imaging Subset of DDSM) dataset is a collection of medical images in JPEG format, derived from the original dataset which was 163GB in size. The resolution of the images in the CBIS-DDSM dataset has been maintained to match that of the original dataset. This dataset is primarily focused on breast imaging for mammography.

### Methodology
1. Data Preprocessing
  * The images are resized to 50x50 pixels and converted to RGB (3 channels).
  * Data augmentation is applied using rotation, width & height shift, shear transform, zoom range, horizontal/vertical flip and fill mode.
  * Dataset split into training and test sets.
  * Labels are one-hot encoded for binary classification (cancer or no cancer). 
2. Model Architecture
 * 4 convolutional layers with batch normalization and max pooling.
 * Fully connected layers with dropout.  
3. Training
 * Adam optimizer
 * BinaryCrossentropy loss function
 * Accuracy metric
 * EarlyStopping stops training if validation loss doesn not improve for 5 epochs.
 * ReduceLROnPlateau reduces learning rate by factor of 0.2 if validation loss stagnates.  
4. Evaluation
 * Loss and accuracy
 * Confusion matrix
 * AUC-ROC curve
 * Precision, recall and f1-score 

**Model Summary:**<br>
<img width="368" alt="image" src="https://github.com/user-attachments/assets/746f2be6-443f-47a8-ab35-22d71d70df88" />

### Evaluation
<img width="407" alt="image" src="https://github.com/user-attachments/assets/e7338bfe-b6f0-4fa0-ba27-48a45b6e39d6" />

![image](https://github.com/user-attachments/assets/d3364a01-0bad-48f8-a3c3-8ee10eb67551)
![image](https://github.com/user-attachments/assets/1917df0b-94c9-4647-81eb-70d1cba8d0dd)<br>
![image](https://github.com/user-attachments/assets/36107b25-9ce3-484b-8514-1b44da834bdd)
![image](https://github.com/user-attachments/assets/8987b527-2e2e-48c3-88f1-6927a50629ec)

## OpenAI Assistants API
### Enemble Learning Model Inference
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

### Image Recognition Inference 
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
