# Brain-Cancer-Prediction-Model

### Overview
This project is focused on the classification of brain tumors using machine learning techniques. Utilizing a dataset of brain MRI images, the goal is to accurately identify and classify different types of tumors. The early and precise diagnosis of brain tumors is crucial, and leveraging machine learning can assist medical professionals in making informed decisions.

### Dataset
The dataset comprises grayscale MRI scans of the brain, each labeled with one of the following categories:
- No Tumor
- Pituitary Tumor
- Glioma Tumor
- Meningioma Tumor

### Methodology
#### Data Preprocessing: 
- MRI images are loaded and resized to a consistent dimension.
- Data augmentation techniques are applied to increase the diversity of the training set and improve model robustness.
- Pixel values are normalized to the range [0, 1].
  
#### Model Training:
A Random Forest Classifier is employed as the primary model. Other machine learning models are also considered and can be experimented with for comparison.
The model is trained on the preprocessed dataset, and its performance is evaluated using a separate testing set.

#### Visualization:
Post prediction, the MRI images from the test set are visualized alongside their predicted labels to provide a visual assessment of the model's accuracy.

#### Evaluation:
The model's accuracy is computed using the test dataset. Additional metrics like F1-score, precision, and recall can be considered for a more comprehensive evaluation, especially if the dataset is imbalanced.

### Results
The project's current iteration achieves an accuracy of 61% using the Random Forest Classifier. Efforts are ongoing to improve this metric by refining the model, tuning hyperparameters, and potentially using more advanced neural network architectures.

### Future Work
- Implement Convolutional Neural Networks (CNNs) to leverage their capacity for handling image data.
- Further data augmentation and preprocessing to enhance the model's generalization capabilities.
- Hyperparameter tuning using techniques like Grid Search or Randomized Search.
- Explore ensemble methods and model combinations for potentially better results.
