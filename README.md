# 🎯 Student Placement Prediction using Machine Learning

## 📌 Overview
This project aims to predict student placement outcomes based on academic performance, technical skills, and experience-related features using supervised machine learning models.

The objective is to analyze key factors influencing placement and build a predictive model that can assist students in understanding their placement readiness.

---

##  Problem Statement
Many students are uncertain about their placement chances and the specific skills they need to improve.

This project addresses that problem by:
- Predicting placement status (Placed / Not Placed)
- Identifying the most influential factors affecting placement

---

##  Dataset Description
The dataset consists of 45,000 student records with the following features:

- Academic: CGPA, Backlogs  
- Skills: Coding Skills, Communication Skills, Soft Skills  
- Experience: Internships, Projects, Certifications  
- Assessment: Aptitude Score  
- Demographics: Age, Gender, Degree, Branch  
- Target Variable: Placement Status  

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Removed irrelevant features (Student_ID)
- Encoded categorical variables using Label Encoding
- Verified data integrity (no missing values)

### 2. Exploratory Data Analysis (EDA)
- Analyzed placement distribution
- Studied relationship between CGPA and placement
- Visualized feature importance

### 3. Model Building
Two classification models were implemented:

- Logistic Regression (baseline model)
- Random Forest Classifier (advanced model)

### 4. Model Evaluation
- Train-test split (80:20)
- Accuracy metric used for evaluation
- Confusion Matrix for detailed performance analysis

---

## Results

| Model | Accuracy |
|------|---------|
| Logistic Regression | 86.4% |
| Random Forest | ~99% |

> Note: High accuracy in Random Forest is due to strong correlation between features and placement outcome.

---

##  Key Insights
- Coding skills and internships are the most influential features
- Higher CGPA significantly improves placement probability
- Combined skill metrics strongly determine outcomes

---

##  Tech Stack
- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

##  Visualizations Included
- Placement Distribution Plot  
- CGPA vs Placement Boxplot  
- Feature Importance Graph  
- Confusion Matrix  

---

## Future Enhancements
- Deploy as an interactive web application (Streamlit)
- Use real-world datasets for better generalization
- Apply advanced models (XGBoost, Neural Networks)

---

## author
AKASH SHUKLA
