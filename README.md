#  Video Game Sales - Data Science Project

 End-to-end Data Science project focused on predicting video game sales and analyzing market segmentation.

---

##  Objective
Develop a machine learning model to predict global video game sales and support business decision-making.

---

## Dataset
- 16,325 records  
- Features:
  - Platform
  - Genre
  - Year
  - Publisher
  - Regional Sales  

---

## Key Challenge: Data Leakage
Initial models showed unrealistically high performance due to the inclusion of regional sales (NA, EU, JP), which directly sum to global sales.

After removing these variables:
- Model performance dropped significantly  
- Revealing the true complexity of the problem  

---

## Models

### Regression
- Random Forest Regressor  
- Linear Regression (baseline)

### Clustering
- KMeans segmentation:
  - Global hits  
  - Regional successes  
  - Low-performing titles  

### Deep Learning
- Neural Network (TensorFlow)  
- EarlyStopping to reduce overfitting  

---

##  Results (After Fixing Leakage)

- RMSE (Random Forest): ~2.10  
- RMSE (Linear Regression): ~1.97  
- R²: ~ -0.03  

👉 Low performance indicates that available features are insufficient to fully explain global sales.

---

##  Key Insights
- Action games dominate globally, while RPGs lead in Japan  
- Sales decline after 2010 (shift toward digital distribution)  
- Year of release is the most influential variable  
- Publisher impact (e.g., Nintendo) significantly affects performance  
- External factors (marketing, reviews, trends) play a key role  

---

##  Limitations
- Lack of data on marketing and user behavior  
- High dimensionality due to one-hot encoding  
- Limited predictive power  

---

##  Future Work
- Feature engineering and dimensionality reduction  
- Integration of external data sources  
- Hyperparameter tuning  
- Model deployment  

---

## 🛠 Tech Stack
- Python (Pandas, Scikit-learn, TensorFlow)  
- Power BI  
- TensorBoard  

---

##  Conclusion
This project highlights the importance of proper data preprocessing, especially avoiding data leakage.

Rather than focusing only on model performance, it emphasizes critical thinking, model validation, and real-world applicability.

---

##  Links
- 📊 Dashboard:   [Click](https://app.powerbi.com/view?r=eyJrIjoiY2NkZjFmZWEtZWM3Ni00YjI5LWFkODQtMGVmZDE4ZWQ4YjcxIiwidCI6IjhiMThhYzYwLWZmNzktNDY4YS1iMDIxLTQ3NmYyOGIyZDU5NyIsImMiOjR9)
- 💻 Notebooks: [Click](https://github.com/mariadvrafo/video-game-data-science/tree/main/notebooks)
