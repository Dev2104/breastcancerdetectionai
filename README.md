# 🧠 Breast Cancer Detection AI

A full-stack machine learning application for early breast cancer prediction using supervised learning models, built with a production-ready pipeline and an interactive Streamlit web interface.

---

## 🚀 Project Overview

## 🌐 Live Demo

🚀 Try the application here:  
👉 https://breastcancerdetectionai-dt2104.streamlit.app

This project predicts whether a breast tumor is **benign or malignant** using diagnostic features from the Breast Cancer Wisconsin dataset.

It is designed as a **research-oriented decision-support system** and demonstrates a complete ML lifecycle:

- Data preprocessing  
- Model training & evaluation  
- Model selection  
- Deployment-ready model packaging  
- Interactive visualization & inference  

---

## 🎯 Objectives

- Build an end-to-end ML pipeline for medical classification  
- Compare multiple supervised learning algorithms  
- Provide interpretable insights using visualizations  
- Deploy a user-friendly web application for predictions  

---

## 🧰 Tech Stack

- **Language:** Python  
- **ML Libraries:** scikit-learn, pandas, numpy  
- **Visualization:** matplotlib, seaborn  
- **Frontend:** Streamlit  
- **Model Persistence:** joblib  

---

## 📁 Project Structure

```
breastcancerdetectionai/
│
├── app/
│   ├── Home.py
│   └── pages/
│       ├── 1_Manual_Prediction.py
│       ├── 2_Batch_Prediction.py
│       ├── 3_Visualizations.py
│       ├── 4_Model_Insights.py
│       └── 5_About_Project.py
│
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── explain.py
│   └── unsupervised.py
│
├── models/
│   ├── best_model.pkl
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── random_forest.pkl
│   ├── support_vector_machine.pkl
│   └── k_nearest_neighbors.pkl
│
├── reports/
│   ├── model_comparison.csv
│
├── requirements.txt
└── README.md
```

---


---

## 🔹 Core Functionality

- Manual prediction using 30 diagnostic features  
- Batch prediction via CSV upload  
- Automatic model selection (best-performing model saved)  
- Prediction probabilities (confidence scores)  

---

## 📊 Model Training & Evaluation

### Models Implemented

- Logistic Regression  
- Decision Tree  
- Random Forest  
- Support Vector Machine  
- K-Nearest Neighbors  

### Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- ROC-AUC  

The system automatically selects the **best-performing model** and saves it in a deployment-ready format.

---

## 📈 Visualizations

### ✅ Implemented

- Class distribution analysis  
- Feature distribution histograms  
- Correlation heatmap  
- Model comparison (accuracy-based)  
- Confusion matrix (saved and displayed)  
- ROC curve  
- PCA (Principal Component Analysis) visualization  
- Feature importance (model interpretability)  

---

## 🔜 Upcoming Features

- Advanced model insights dashboard  
- SHAP / XAI-based interpretability (Explainable AI)  
- PCA variance explanation visualization  
- Model performance comparison (multi-metric dashboard)  
- Downloadable prediction reports  

- 📄 **Health Report PDF Generation (Samsung Health–inspired)**  
  - Generate professional medical-style reports for predictions  
  - Include:
    - Patient input summary  
    - Prediction result (benign/malignant)  
    - Confidence score  
    - Key influencing features  
    - Visual charts (probability, feature impact)  
  - Export as downloadable PDF for real-world usability  

- Model retraining interface from UI  
- User input validation & error handling improvements  
- Enhanced UI/UX for professional deployment  

---

## 🌐 Web Application

The application is built using **Streamlit** and includes:

- 🏠 Home Page → Project overview  
- ✍️ Manual Prediction → Individual input prediction  
- 📂 Batch Prediction → CSV upload predictions  
- 📊 Visualizations → Data + model insights  

---

## ⚙️ How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/breastcancerdetectionai.git

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
