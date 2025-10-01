# 🚀 Enhanced AutoML Dashboard

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)]()
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-green)]()

A comprehensive web application for automated machine learning, data exploration, and visualization.  

</div>

---

## 📋 Overview
The **Enhanced AutoML Dashboard** is a powerful **Streamlit-based web application** that provides end-to-end **automated machine learning capabilities**.  
From **data loading and exploration** to **model training and deployment**, this dashboard offers an **intuitive interface** for both beginners and experienced data scientists.  

![AutoML Dashboard Preview](https://via.placeholder.com/800x400/667eea/ffffff?text=AutoML+Dashboard+Preview)

---

## ✨ Features

### 🔍 Exploratory Data Analysis (EDA)
- **Data Overview**: Shape, memory usage, duplicate analysis  
- **Statistical Summary**: Descriptive stats for numerical & categorical features  
- **Pattern Analysis**: Correlation matrices, relationship insights  
- **Outlier Detection**: Automated identification + visualization  
- **Data Quality Report**: Full assessment of dataset  
- **Automated Insights**: Smart recommendations  

### 🧹 Advanced Data Cleaning
- **Missing Value Treatment**: Mean, median, mode, forward/backward fill  
- **Data Type Conversion**: Auto-detection & conversion  
- **Outlier Handling**: Removal, capping, transformations  
- **Quality Scoring**: Automated quality checks  
- **Interactive Cleaning**: Real-time preview of changes  

### 📊 Enhanced Visualization
- **Multiple Chart Types**: 15+ options (histogram, scatter, box, violin, etc.)  
- **Interactive Plots**: Powered by **Plotly**  
- **Themes**: Multiple color themes and styling options  
- **Advanced Charts**: 3D scatter, bubble charts, treemaps, swarm plots  
- **Export Capabilities**: Save plots in multiple formats  

### 🤖 Intelligent Model Training
- **Auto Problem Detection**: Classification vs. Regression auto-detection  
- **Supported Algorithms**:  
  - *Classification*: Logistic Regression, Random Forest, Decision Tree, Gradient Boosting, SVM, KNN, AdaBoost  
  - *Regression*: Linear Regression, Random Forest, Decision Tree, Gradient Boosting, SVR, Ridge, Lasso, KNN, AdaBoost  
- **Smart Preprocessing**: Feature encoding, scaling  
- **Hyperparameter Tuning**: GridSearchCV integration  
- **Cross-Validation**: K-fold with metrics  
- **Model Comparison**: Performance metrics visualization  

### 🔧 Label Encoding Management
- **Preserves Original Values** while encoding  
- **Encoder Storage** for consistent transformation  
- **Mapping Display** to users  
- **Seamless Prediction** with auto-encoding  

### 🧪 Advanced Model Testing
- **Interactive Prediction**: Easy input forms  
- **Batch Prediction**: Upload CSV/Excel for multiple predictions  
- **Probability Display**: Class probabilities for classifiers  
- **Uncertainty Estimation**: Prediction intervals for regression  
- **Export Results**: Download predictions as CSV  

---

## 🛠 Installation

### Prerequisites
- Python **3.8+**  
- `pip` package manager  

### Steps
```bash
Clone the repository
git clone https://github.com/your-username/Enhanced-AutoML-Dashboard.git

# Navigate into the project
cd Enhanced-AutoML-Dashboard

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

🎮 Usage

Upload your dataset (CSV/Excel)

Explore dataset with EDA & visualization

Clean & preprocess using interactive options

Train multiple ML models automatically

Compare results & download best model

Make real-time or batch predictions

💡 Future Improvements

Auto feature engineering

Deep learning integration

Model explainability with SHAP

Cloud deployment support
