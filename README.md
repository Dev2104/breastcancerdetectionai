# 🩺 Breast Cancer Detection AI

A machine learning-powered web application that predicts whether a breast tumor is **benign** or **malignant** using clinical diagnostic features.

---

## 📌 Project Overview

Breast cancer is one of the most common cancers worldwide. Early detection significantly improves treatment outcomes.

This project builds an **end-to-end machine learning pipeline** combined with an **interactive web interface** to assist in early detection by analyzing tumor characteristics.

The system:

* Processes clinical diagnostic data
* Trains multiple machine learning models
* Selects the best-performing model
* Provides predictions through a user-friendly web application

---

## 🚀 Features

- 🏠 Home dashboard with model overview
- 🔍 Manual single prediction
- 📂 Batch prediction via CSV upload
- 📊 Model insights and visualization (in progress)
- 🧠 Modular ML pipeline (training, preprocessing, prediction)
- 🌐 Streamlit multipage architecture

## 🔹 Core Functionality

- Manual prediction using 30 diagnostic features  
- Batch prediction via CSV upload  
- Automatic model selection (best-performing model saved)  
- Prediction probabilities (confidence scores)  

### 🚀 Advanced Features

- Interactive Streamlit multipage application  
- Model insights and feature importance visualization (upcoming)  
- Advanced data visualizations and exploratory analysis (upcoming)  
- Explainable AI integration for model interpretability (upcoming)  
- Clinical-grade structured PDF report generation with diagnostic summary and probability interpretation (upcoming)  
- Risk scoring and severity interpretation system (upcoming)  
- Enhanced UI/UX with dashboard-style layout (upcoming)  

### 🔹 Web Application

* Built with Streamlit
* Multi-page navigation:

  * Home
  * Manual Prediction
  * Batch Prediction
  * Visualizations (planned)
  * Model Insights (planned)
  * About Project

### 🔹 Machine Learning

* Models implemented:

  * Logistic Regression
  * Decision Tree
  * Random Forest
  * Support Vector Machine (SVM)
  * K-Nearest Neighbors (KNN)
* Automatic scaling for models that require it
* Model comparison based on accuracy
* Deployment-ready model bundle

---

## 🧠 How It Works

1. **Data Loading**

   * Dataset is loaded and cleaned

2. **Preprocessing**

   * Train-test split
   * Feature scaling (StandardScaler)

3. **Model Training**

   * Multiple models are trained
   * Each model is evaluated on test data

4. **Model Selection**

   * Best model selected based on accuracy

5. **Model Saving**

   * Saved as a bundle including:

     * model
     * scaler
     * feature names
     * metadata

6. **Prediction**

   * Input is validated
   * Scaling applied if required
   * Prediction + probability returned

---

## 🏗️ Project Structure

```bash
## 📁 Project Structure

breastcancerdetectionai/
│
├── app/
│   └── pages/
│       ├── 1_Manual_Prediction.py      # Manual input prediction UI
│       ├── 2_Batch_Prediction.py       # CSV batch prediction UI
│       ├── 3_Visualizations.py         # Data & model visualizations
│       ├── 4_Model_Insights.py         # Model explainability & insights
│       ├── 5_About_Project.py          # Project overview & methodology
│       └── Home.py                     # Home / landing page (main UI)
│
├── data/                               # Dataset storage
├── models/
│   └── best_model.pkl                  # Trained model bundle
├── notebooks/
│   └── eda_and_modeling.ipynb          # EDA & experimentation
├── reports/                            # Generated outputs (future PDFs)
│
├── src/
│   ├── data_loader.py                  # Data loading logic
│   ├── preprocess.py                   # Preprocessing pipeline
│   ├── train.py                        # Model training logic
│   ├── predict.py                      # Prediction pipeline
│   ├── explain.py                      # Model explainability
│   └── unsupervised.py                 # Optional unsupervised analysis
│
├── venv/                               # Virtual environment (ignored in Git)
├── main.py                             # Optional execution script
├── requirements.txt                    # Dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/breastcancerdetectionai.git
cd breastcancerdetectionai
```

### 2. Create virtual environment

```bash
python -m venv venv
```

### 3. Activate environment

```bash
venv\Scripts\activate   # Windows
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```bash
streamlit run app/streamlit_app.py
```

Then open:

👉 http://localhost:8501

---

## 📊 Example Output

* Prediction: **Malignant / Benign**
* Numeric Output: `0` or `1`
* Probability:

  * Malignant: 0.98
  * Benign: 0.02

---

## 📈 Future Improvements (Thesis Enhancements)

* Data visualization dashboard
* Feature importance (SHAP / explainability)
* Hyperparameter tuning
* Cross-validation metrics
* Unsupervised learning (clustering, PCA)
* Model comparison dashboard
* Deployment (Streamlit Cloud / AWS)

---

## ⚠️ Disclaimer

This application is intended for **educational and research purposes only**.

It is **not a medical diagnostic tool** and should not be used for real-world medical decisions.

---

## 👨‍💻 Author

**Dev Tailor**
M.Sc. Data Science Student
Germany 🇩🇪

---

## ⭐ If you found this project useful

Give it a star ⭐ on GitHub — it helps a lot!
