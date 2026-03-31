# 🧠 Breast Cancer Detection AI

🚀 **Live App:** https://breastcancerdetectionai-dt2104.streamlit.app/

An end-to-end AI-powered system for early breast cancer detection using machine learning and an interactive multi-page Streamlit interface.

---

## 📌 Overview

This project is a complete machine learning pipeline designed to simulate a real-world medical diagnostic system for breast cancer detection. It includes data preprocessing, model training, evaluation, deployment, and an interactive user interface.

The goal is to build a **thesis-level AI application** that not only predicts cancer but also evolves toward explainability, reporting, and real-world usability.

---

## 🔹 Core Functionality

* Manual prediction using 30 diagnostic features
* Batch prediction via CSV upload
* Automatic model selection (best-performing model saved)
* Prediction probabilities (confidence scores)
* Interactive multi-page Streamlit UI (Home, Prediction, Insights, Visualizations)

### 🔜 Upcoming Features

* PDF medical-style report generation (like ECG reports)
* Model explainability using SHAP / feature importance
* Advanced evaluation metrics (ROC-AUC, confusion matrix, precision-recall)
* Patient history tracking (simulation)
* Risk scoring system beyond binary classification
* Model comparison dashboard
* Export predictions as downloadable reports

---

## 📂 Project Structure

```
breastcancerdetectionai/
│
├── app/
│   ├── Home.py
│   ├── pages/
│   │   ├── 1_Manual_Prediction.py
│   │   ├── 2_Batch_Prediction.py
│   │   ├── 3_Visualizations.py
│   │   ├── 4_Model_Insights.py
│   │   ├── 5_About_Project.py
│
├── data/
├── models/
│   └── best_model.pkl
│
├── notebooks/
│   └── eda_and_modeling.ipynb
│
├── reports/
│
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   ├── explain.py
│   ├── unsupervised.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Run Locally

```bash
# Clone the repository
git clone https://github.com/Dev2104/breastcancerdetectionai.git

# Navigate to project
cd breastcancerdetectionai

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/Home.py
```

---

## ☁️ Deployment

This application is deployed on **Streamlit Community Cloud**.

* **Main entry file:** `app/Home.py`
* The app automatically rebuilds on every GitHub push

---

## 🤖 Models Used

* Logistic Regression
* Decision Tree
* Random Forest
* Support Vector Machine (SVM)
* K-Nearest Neighbors (KNN)

The best-performing model is automatically selected and saved for deployment.

---

## 📊 Dataset

* Breast Cancer Wisconsin Dataset (from sklearn)
* 30 numerical features extracted from digitized tumor images
* Target:

  * `0 → Malignant`
  * `1 → Benign`

---

## 📈 Future Scope

This project is being extended toward a **real-world clinical AI system**, including:

* AI-powered medical reports
* Explainable AI (XAI)
* Clinical-style dashboards
* Multi-model ensemble learning
* Integration with external datasets

---

## ⚠️ Disclaimer

This application is for **educational and research purposes only**.
It is **not intended for medical diagnosis or clinical use**.

---

## 💼 Author

**Dev Tailor**
Aspiring Data Scientist | AI & Analytics Enthusiast

This project is part of my Master's thesis focused on building real-world AI applications.

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!
