# Deep Learning Approaches to Enhance Neurogenesis Research

## Overview
This project explores the application of advanced deep learning techniques, specifically Convolutional Neural Networks (CNNs), to improve the analysis of neuroimaging data related to neurogenesis, particularly in the context of Alzheimer's disease. By leveraging automated techniques, the proposed system aims to provide more accurate and reliable predictions of neurogenesis rates compared to traditional methods.

## Objectives
- Develop a robust system for neurogenesis analysis.
- Identify factors influencing neurogenesis.
- Integrate the system into existing research frameworks.
- Enhance understanding of brain plasticity and contribute to developing targeted treatments for neurological conditions.

## Methodology
1. **Data Collection**: Collect neuroimaging data from publicly available datasets such as the Alzheimer MRI Dataset from Kaggle.
2. **Data Preprocessing**: Preprocess the images for normalization and augmentation to improve model training.
3. **Model Development**: 
   - Implement data augmentation techniques to enhance model robustness.
4. **Model Evaluation**: Assess model performance using metrics such as accuracy, precision, recall, and F1-score.

## Results
The proposed model achieved an overall accuracy of 92% in classifying neurogenesis among different patient groups. The findings indicate that accurate classification can serve as a valuable biomarker for monitoring disease progression in Alzheimer's patients.

## Key Insights
- There is a strong correlation between lower neurogenesis rates and cognitive decline in Alzheimer's patients.
- The ability to classify neurogenesis rates accurately positions this metric as a potential biomarker for tracking disease progression.
- Data augmentation techniques significantly improved model robustness and generalization capabilities.

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/AdityaNPatil/deep-learning-neurogenesis.git
   cd deep-learning-neurogenesis
   ```

2. Install required packages

3. Ensure you have TensorFlow installed (version 2.x recommended):
   ```bash
   pip install tensorflow
   ```

## Usage
To train the model, run the following command:
```bash
python train.py
```
This will start the training process using the specified dataset and parameters defined in `train.py`.

To evaluate the model, use:
```bash
python evaluate.py
```

To start app:
```bash
python uploadTemp.py
```

## Future Directions
- Incorporate multimodal data (e.g., combining MRI with PET scans) to improve model accuracy.
- Explore more advanced architectures, such as attention mechanisms, to capture spatial relationships in the brain more effectively.
- Conduct longitudinal studies to assess the model's predictive power for disease progression.

## Contact Information
For any inquiries or feedback regarding this project, please contact:
- Aditya Patil: adityapatil2708@gmail.com
- Rupesh Dahibhate: rupeshdahibhate2003@gmail.com

Feel free to modify any section or add additional information relevant to your project! If you need further assistance or specific changes made, let me know!

Datasets:
- IXI: [Here](https://brain-development.org/ixi-dataset/) or [Here](https://www.nitrc.org/ir/app/action/ProjectDownloadAction/project/ixi)
- Kaggle: [Here](https://www.kaggle.com/datasets/borhanitrash/alzheimer-mri-disease-classification-dataset)

