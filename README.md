# 🎮 Video Game Sales - Data Science Project

##  Objective
Develop a machine learning model to predict global video game sales and segment the market to support business decision-making.

---

## Dataset
- 1616.325 records
- Features:
  - Platform
  - Genre
  - Year
  - Publisher
  - Regional Sales

---

##  Data Processing
- Missing values removal
- One-hot encoding of categorical variables
- Feature selection to avoid data leakage

---

##  Key Challenge: Data Leakage

Initial models showed extremely high performance due to the inclusion of regional sales (NA, EU, JP), which directly sum to global sales.

After removing these variables:
- The model performance dropped significantly
- Revealing the true complexity of the problem

---

##  Models

###  Regression
- Random Forest Regressor  
- Linear Regression (baseline)

###  Clustering
- KMeans for market segmentation:
  - Global hits
  - Regional successes
  - Low-performing titles

###  Deep Learning
- Neural Network with TensorFlow
- EarlyStopping applied to reduce overfitting

---

## Results (After Fixing Leakage)

- RMSE: ~2.10 
- RMSE Linear:~1.97
- R²: ~ -0.03  

=> The low performance indicates that available features are insufficient to fully explain sales.

---

##  Key Insights

- Action games dominate global sales, while RPGs are more prominent in Japan
- Sales show a decline after 2010 (likely due to shift toward digital distribution)
- The year of release is the most influential variable
- Publisher impact (e.g., Nintendo) significantly affects performance
- Sales are influenced by external factors not present in the dataset (marketing, reviews, trends)

---

##  Limitations

- Lack of data on marketing, user ratings, and digital sales
- High dimensionality due to one-hot encoding
- Limited predictive power with available features

---

##  Future Work

- Feature engineering (grouping categories, reducing dimensionality)
- Incorporation of external data sources
- Hyperparameter tuning
- Deployment of model for real-time predictions

---

##  Tech Stack

- Python (Pandas, Scikit-learn, TensorFlow)
- Power BI
- TensorBoard

---

##  Conclusion

This project demonstrates the importance of proper data preprocessing, especially avoiding data leakage, and highlights the challenges of building predictive models with limited features.

Rather than focusing only on model performance, this work emphasizes critical thinking, model validation, and real-world applicability.
