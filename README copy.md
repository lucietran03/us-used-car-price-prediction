# 🚗 Machine Learning for Car Price Prediction

## 📋 Overview

This project focuses on predicting the prices of used cars using machine learning models. The dataset contains various features related to car specifications, such as engine performance, fuel economy, legroom, and seller ratings. The goal is to build, evaluate, and fine-tune machine learning models to accurately predict car prices and gain insights into the factors influencing these prices.

---

## 📊 **1. Dataset**

- The dataset used in this project is sourced from Kaggle.

### 🔗 **Reference**

- **Dataset URL**: [US Used Cars Dataset](https://www.kaggle.com/datasets/ananaymital/us-used-cars-dataset?select=used_cars_data.csv)
- **Date Accessed**: August 23, 2024

---

## 📈 **2. Folder: Figures**

This folder contains key visualizations and graphs generated throughout the project.

### 📂 **Contents**

- **Data Exploration Figures**: Histograms, scatter plots, and box plots for EDA.
- **Correlation Matrix**: Heatmaps showing relationships between features.
- **Model Performance Visualizations**:
  - Residual plots to analyze model errors.
  - Feature importance plots for tree-based models.
  - Cross-validation performance trends.
- **Hyperparameter Tuning Results**: Visual comparisons of different hyperparameter values and their effects on model performance.

This folder ensures that all important figures are saved for future reference and reporting.

---

## 🤖 **3. Folder: Models**

This folder stores trained machine learning models for used car price prediction.

### 📂 **Contents**

- **Trained Models**: Serialized versions of the trained models using `joblib` or `pickle`, allowing easy reloading for inference.
- **Model Performance Reports**: JSON or text files summarizing the evaluation metrics (R², MAE, RMSE) for each trained model.
- **Checkpoint Files**: Saved intermediate models during training, useful for resuming long-running processes.

### 🗂️ **Example Files**

- `ridge_regression.pkl` - Saved Ridge Regression model.
- `random_forest_model.pkl` - Trained Random Forest Regressor.
- `gradient_boosting_results.json` - Performance metrics for Gradient Boosting.

This setup allows for reproducibility and easy deployment of the trained models.

---

## 💾 **4. Folder: Saved_Objects**

This folder stores various intermediate objects required during the project.

### 📂 **Contents**

- **Preprocessed Data Files**:
  - `processed_train.csv` – Cleaned and transformed training data.
  - `processed_test.csv` – Preprocessed test data for evaluation.
- **Feature Transformation Objects**:
  - `scaler.pkl` – StandardScaler/MinMaxScaler object used for normalizing numerical features.
  - `encoder.pkl` – OneHotEncoder or LabelEncoder used for categorical features.
- **Cross-Validation Results**:
  - `kfold_results.json` – Stores results of K-Fold Cross-Validation for different models.

This folder ensures that important objects are saved separately for reuse and avoids the need for reprocessing the raw data multiple times.

---

## 🐍 **5. Folder: Python 3.12.4**

This folder contains scripts and configurations required for running the Jupyter Notebook and machine learning pipeline.

### 📂 **Contents**

- **Python Virtual Environment Files**: If a virtual environment is used, this folder may contain installed dependencies.
- **Script Files**:
  - `data_preprocessing.py` – Standalone script for handling data cleaning and feature engineering.
  - `train_model.py` – Script for training different machine learning models.
  - `evaluate_model.py` – Script for running model evaluation and generating performance metrics.
- **Requirements File**:
  - `requirements.txt` – List of required Python libraries (e.g., `pandas`, `scikit-learn`, `matplotlib`).

This structure ensures that the project can be easily run in a standardized Python environment.

---

## 🎯 **6. Objectives**

1. **Data Exploration and Preprocessing**:

   - Analyze the dataset to understand its structure and identify key features.
   - Handle missing values, outliers, and noisy data to ensure data quality.
   - Encode categorical features and transform numerical features for better model performance.

2. **Model Training and Evaluation**:

   - Train multiple machine learning models, including Ridge Regression, RandomForestRegressor, and GradientBoostingRegressor.
   - Evaluate the models using metrics such as R² score, Mean Absolute Error (MAE), and Root Mean Square Error (RMSE).
   - Use K-fold cross-validation to ensure the models generalize well to unseen data.

3. **Hyperparameter Tuning**:

   - Fine-tune the hyperparameters of the models using RandomizedSearchCV to optimize their performance.
   - Justify the chosen hyperparameter ranges and their impact on the models.

4. **Model Testing and Insights**:

   - Test the best-performing models on a separate test dataset.
   - Analyze feature importance to understand which factors most influence car prices.
   - Compare the performance of the models and select the most suitable one.

5. **Documentation and Reporting**:
   - Document the entire workflow, including data preprocessing, model training, evaluation, and insights.
   - Provide a detailed discussion of the results, limitations, and areas for improvement.

---

## 🔑 **7. Key Findings**

- **Best Model**: RandomForestRegressor achieved the highest accuracy with an R² score of 0.9 and the lowest RMSE, making it the most suitable model for predicting used car prices.
- **Feature Importance**: Features such as front_legroom, back_legroom, width, and seller_rating were identified as the most influential factors affecting car prices.
- **Limitations**: Challenges such as noisy data, missing values, and real-world external factors were identified as potential limitations to the model's performance.

---

## 🏁 **8. Conclusion**

This project demonstrates the application of machine learning techniques to predict used car prices effectively. By leveraging data preprocessing, model evaluation, and hyperparameter tuning, the project provides valuable insights into the factors influencing car prices and highlights the importance of selecting the right model for the task.

---

## 📚 **9. References**

A list of references is included to credit the sources used for the project, including datasets, documentation, and external research articles.

---

## 🤝 **10. Project Contribution**

- **👩‍💻 Tran Dong Nghi** - Team Leader, Data Loading, Model Training, K-fold Validation, Hyperparameter Tuning, Model Testing

- **📊 Vo Thuy Khanh Ngoc** - Data Visualization, Outlier Handling, Categorical Feature Encoding, Model Selection, Model Performance Analysis

- **🔍 Ho Thanh Hoa** - Missing Values Handling, Model Evaluation

- **📈 Chan Yong Park** - Data Insights, K-fold Validation Definition, Hyperparameter Justification, Model Performance Evaluation

---

## 🏆 **11. Conclusion**

This project demonstrates the use of machine learning to predict used car prices, with RandomForestRegressor identified as the most effective model. Key insights, such as the impact of mileage, brand, and engine power, provide valuable guidance for stakeholders. Challenges like noisy data and overfitting were addressed, but future work could explore external factors and advanced algorithms to enhance accuracy and insights. This project underscores the potential of data-driven approaches in the used car market.

## 🛠️ **12. How to Run the Project**

### Prerequisites

1. Install Python 3.12.4 or higher.

### Steps to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/us-used-car-price-prediction.git
cd us-used-car-price-prediction
```

2. Preprocess the data:

```bash
python scripts/data_preprocessing.py
```

3. Train the models:

```bash
python scripts/train_model.py
```

4. Evaluate the models:

```bash
python scripts/evaluate_model.py
```

5. View the results and visualizations in the `Figures` folder.

---

## 📞 **13. Contact**

For any questions or collaboration opportunities, feel free to reach out:

- **Tran Dong Nghi**: [trandongnghi05@gmail.com](mailto:trandongnghi05@gmail.com)
- **Vo Thuy Khanh Ngoc**
- **Ho Thanh Hoa**
- **Chan Yong Park**

Alternatively, open an issue on the [GitHub repository](https://github.com/your-username/us-used-car-price-prediction/issues).
