# Student Dropout Prediction App 🎓

## Overview
The **Student Dropout Prediction App** is a machine learning solution designed to identify students at risk of dropping out.  
It combines interpretable models, risk scoring, and visual storytelling to empower advisors and educators with actionable insights.

## Features
- 📊 **Data Exploration**: Descriptive statistics and visualizations of student demographics and performance.
- 🤖 **Model Training**: Logistic Regression, Random Forest, XGBoost, SVM, and LSTM models compared for accuracy and interpretability.
- 🧩 **Modular Pipeline**: Clean, maintainable code for preprocessing, training, and evaluation.
- 🌐 **Streamlit Dashboard**: Interactive interface for advisors to explore predictions and risk scores.
- 🔍 **Interpretability**: Feature importance, SHAP values, and transparent scoring to support decision-making.

## Project Structure

Student_dropout_app/ │── data/                # Raw and processed datasets 
│── notebooks/           # Colab/Jupyter notebooks for exploration 
│── src/                 # Modular ML pipeline code 
│── app/                 # Streamlit dashboard scripts 
│── results/             # Model outputs, metrics, and visualizations 
│── README.md            # Project documentation

  
## Installation
Clone the repository:
```bash
git clone https://github.com/your-username/Student_dropout_app.git
cd Student_dropout_app

  ###Install dependencies:
  pip install -r requirements.txt

  ###Usage
Run the Streamlit app:
  streamlit run app/dashboard.py

  Open your browser at http://localhost:8501 to interact with the dashboard.
Dataset
We use the xAPI-Edu-Data dataset, which contains student demographics, academic performance, and engagement features.
Dataset link (archive.ics.uci.edu in Bing)
Results
- Logistic Regression: Baseline interpretability
- Random Forest & XGBoost: Strong performance with feature importance
- LSTM: Sequential modeling of student engagement
- Final dashboard integrates risk scoring and visual explanations

Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.
License
This project is licensed under the MIT License.

---

## 📝 Colab Notebook Descriptions

### At the **Top of the Notebook**
```markdown
# Student Dropout Prediction – Colab Notebook 🎓

This notebook walks through the complete pipeline for predicting student dropout risk.  
We will:
- Load and explore the xAPI-Edu-Data dataset
- Perform preprocessing (encoding, scaling, cleaning)
- Train multiple machine learning models (Logistic Regression, Random Forest, XGBoost, SVM, LSTM)
- Compare performance metrics and interpretability
- Prepare modular code for deployment in the Streamlit app

###Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.
###License
This project is licensed under the MIT License.







  
  
