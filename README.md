# AI-Based Symptom Checker and Wellness Recommender for Early Detection and Lifestyle Guidance in PCOS Management

This project presents an **AI-driven system for early detection and holistic management of Polycystic Ovary Syndrome (PCOS)**.  
The solution integrates **machine learning-based prediction**, **REST API deployment**, and **Android mobile application connectivity** to support accurate diagnosis and personalized lifestyle guidance.

The work is supported by an extensive literature survey and a **peer-reviewed publication in a UGC Care Group-II journal**.

---

###  App Interface Screens

| Home Screen | User Input Screen |
|------------|------------------|
| ![](./app_screens/app1.png) | ![](./app_screens/app5.png) |

| Prediction Result | Lifestyle Recommendation |
|------------------|--------------------------|
| ![](./app_screens/app6.png) | ![](./app_screens/app7.png) |

| API Connectivity | System Output |
|------------------|--------------|
| ![](./app_screens/app8.png) | ![](./app_screens/app9.png) |

---

###  Additional Application Screens

| Screen 10 | Screen 11 |
|----------|-----------|
| ![](./app_screens/app10.png) | ![](./app_screens/app11.png) |

| Screen 12 | Screen 13 |
|----------|-----------|
| ![](./app_screens/app12.png) | ![](./app_screens/app13.png) |

| Screen 14 | Screen 15 |
|----------|-----------|
| ![](./app_screens/app14.png) | ![](./app_screens/app15.png) |

| Screen 16 | Screen 17 |
|----------|-----------|
| ![](./app_screens/app16.png) | ![](./app_screens/app17.png) |

| Screen 18 | Screen 19 |
|----------|-----------|
| ![](./app_screens/app18.png) | ![](./app_screens/app19.png) |

---

##  Project Objectives

- To collect, preprocess, and analyze clinical data related to PCOS
- To identify key diagnostic features influencing PCOS prediction
- To develop and compare machine learning models for accurate detection
- To deploy a cloud-based prediction API
- To integrate prediction results with an Android mobile application
- To provide lifestyle guidance for effective PCOS management

---

##  Literature Survey

- Reviewed **30+ AI-based research papers** on PCOS diagnosis and management
- Common algorithms studied:
  - Random Forest (RF)
  - Support Vector Machine (SVM)
  - XGBoost
  - CNN-based ultrasound models
- Datasets analyzed:
  - Clinical and hormonal datasets
  - Electronic Health Records (EHR)
  - Gene expression and imaging data
- Reported accuracy range: **85% – 99.31%**
- Recent trends:
  - Explainable AI (SHAP, LIME)
  - Ensemble learning
  - Mobile and cloud-based deployment

### Identified Research Gaps
- Lack of unified diagnosis + lifestyle platforms
- Limited real-world validation
- Low interpretability and clinical trust

---

##  Problem Statement

PCOS remains significantly underdiagnosed due to overlapping symptoms and heterogeneous diagnostic criteria.  
Existing systems focus either on diagnosis or lifestyle management, lacking integration and explainability.  
There is a strong need for a **transparent, AI-driven, and user-centric platform** enabling early detection and personalized guidance.

---

##  Proposed Solution

The AI-PCOS system provides:

- **Intelligent Diagnosis:**  
  ML-based analysis of clinical and lifestyle parameters
- **Personalized Wellness Guidance:**  
  Diet, exercise, and habit recommendations
- **Android Mobile Application:**  
  User-friendly interface for real-time prediction
- **Cloud-Based API:**  
  Scalable backend for prediction services
- **Future-Ready Design:**  
  Supports Explainable AI and advanced analytics

---

##  System Architecture

1. User enters health parameters in Android application  
2. Data is sent to the cloud-hosted Flask API  
3. Machine learning model processes the input  
4. Prediction result is generated  
5. Result is returned to the mobile application  

---

##  Live Prediction API

**Deployed API Endpoint (Used in Android App):**

https://pcosapi-eq6x.onrender.com/predict


### Android API Integration
```java
URL url = new URL("https://pcosapi-eq6x.onrender.com/predict");

Communication via HTTPS

JSON-based request and response

API tested using Postman

---

## Technology Stack

- **Programming Language:** Python  
- **Backend Framework:** Flask  
- **Machine Learning:** Scikit-Learn  
- **API Testing:** Postman  
- **Mobile Platform:** Android  
- **Deployment:** Render Cloud  

---

##  Installation & Execution

```bash
git clone https://github.com/Radhika5156/pcosapi.git
cd pcosapi
python -m venv venv
pip install -r requirements.txt
python app.py

## Result Analysis & Implementation

Machine learning models evaluated on processed datasets

Best-performing model deployed as a REST API

API validated using Postman

End-to-end integration tested with Android application

System provides reliable and real-time prediction results

## Project Development Timeline

Literature review and problem identification

Data preprocessing and feature analysis

Model training and evaluation

API development and deployment

Android app integration

Testing and documentation

### Research Publication

Title: A Comprehensive Review of AI-Based Approaches for PCOS Diagnosis
Journal: Degres Journal (UGC Care Group-II)
ISSN: 0376-8163
Volume & Issue: Volume 10, Issue 10
Year: 2025

Authors:
Radhika Vyas, Apeksha Patil, Rushikesh Patil, Mayank Sohani

🔗 https://degres.eu/volume-10-issue-10-2025/

# Project Team

Radhika Vyas

Apeksha Patil

Rushikesh Patil

Project Guide:
Dr. Mayank Sohani

Department of Computer Science
SVKM’s NMIMS MPSTME, Shirpur Campus
Academic Year: 2025–26

## Future Enhancements

CNN-based ultrasound image diagnosis

Explainable AI (SHAP / LIME)

Wearable device integration

Secure data storage mechanisms

Doctor and community support modules
