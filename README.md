# Diabetic Retinopathy Detection using CNN

This project aims to classify fundus images into different severity levels of diabetic retinopathy (0 - No DR, 1 - Mild, 2 - Moderate, 3 - Severe, 4 - Proliferative DR) using a convolutional neural network (CNN). The dataset is sourced from a Kaggle competition.

---

## Tools and Frameworks

- **Programming Language:** Python
- **Deep Learning Framework:** TensorFlow/Keras (or PyTorch)
- **Libraries:** NumPy, Pandas, Matplotlib, Scikit-learn, OpenCV
- **Model:** best_resnet50_model.keras

---

## Project Steps

### 1. Data Acquisition and Preparation

- **Download Data:** Obtain `train.zip`, `trainLabels.csv`, and `test.zip` from the Kaggle competition page. Download `sampleSubmission.csv` for prediction formatting.
- **Unzip and Organize:** Extract and organize images into subfolders corresponding to their severity levels based on `trainLabels.csv`.
- **Load Labels:** Use Pandas to read `trainLabels.csv` containing image names and their labels.

### 2. Image Preprocessing

- **Resizing:** Standardize image sizes (e.g., 224x224 or 299x299).
- **Normalization:** Normalize pixel values to the range [0, 1].
- **Data Augmentation:** Enhance dataset with techniques like:
  - Random rotations
  - Horizontal/vertical flips
  - Brightness and contrast adjustments

### 3. Data Splitting

- **Train/Validation Split:** Divide data (e.g., 80% training, 20% validation) using a stratified approach to maintain class distribution.
- **Test Data:** Use the unlabeled test set for predictions after training.

### 4. CNN Model Selection and Design

- **Transfer Learning:** Use a pre-trained model as a base:
  - Options: ResNet50, InceptionV3, EfficientNet, Xception
- **Custom Layers:**
  - Add a global average pooling layer, dense layers with ReLU activation, dropout layers, and an output layer with softmax activation.

### 5. Model Training

- **Loss Function:** Categorical cross-entropy.
- **Optimizer:** Adam or RMSprop with an initial learning rate (e.g., 1e-4).
- **Metrics:**
  - Accuracy
  - Precision, Recall, F1-score (for imbalanced data)
  - Quadratic Weighted Kappa (official Kaggle metric)
- **Training Procedure:**
  - Freeze pre-trained layers initially.
  - Train custom layers.
  - Optionally fine-tune pre-trained layers with a smaller learning rate.

- **Callbacks:**
  - `ModelCheckpoint` for saving best weights.
  - `EarlyStopping` to prevent overfitting.
  - `ReduceLROnPlateau` for learning rate adjustment.

### 6. Model Evaluation

- **Validation Set Metrics:**
  - Accuracy, Precision, Recall, F1-score.
  - Confusion matrix analysis.
  - Quadratic Weighted Kappa calculation.
- **Visualization:** Plot ROC curves and AUC for each class.

### 7. Addressing Class Imbalance

- **Techniques:**
  - Data augmentation targeted at minority classes.
  - Class weights in training (`class_weight` in Keras).
  - Oversampling/undersampling (e.g., SMOTE).

### 8. Model Deployment and Prediction

- **Test Predictions:**
  - Load the best model and predict severity levels for test images.
  - Save predictions in the format of `sampleSubmission.csv`.
- **Submit:** Upload predictions to Kaggle for evaluation.

### 9. Further Improvements

- **Ensemble Learning:** Combine multiple models for better performance.
- **Hyperparameter Tuning:** Experiment with grid/random search or Bayesian optimization.
- **Advanced Preprocessing:** Techniques like CLAHE or Gaussian filtering for better image quality.

---

## Ethical Considerations

AI in medical diagnosis requires thorough validation. This project is for research and educational purposes only. Always consult medical professionals for diagnosis and treatment.

---

## Computational Resources

Training deep learning models requires a GPU. Platforms like Google Colab or Kaggle Kernels can be used for faster training.

---

## Contribution

Contributions are welcome! Please submit issues or pull requests to enhance the project.

---

## License

This project is for educational purposes under the [MIT License](LICENSE).
