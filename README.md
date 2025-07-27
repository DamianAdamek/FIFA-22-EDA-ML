# 🎮 FIFA 22 Player Analysis Using Machine Learning

This project was developed as part of the **Systems and Decision Methods** course. It focuses on exploratory data analysis (EDA) of FIFA 22 player data and demonstrates various machine learning models for predicting player value.

**Author**: Damian Adamek  
**Student ID**: 280572  
**Repository**: [github.com/DamianAdamek/FIFA-22-EDA-ML](https://github.com/DamianAdamek/FIFA-22-EDA-ML)

---

## 🧾 Project Overview

The dataset contains over 19,000 FIFA 22 players, each described by 110 attributes. For analysis, 20 key numerical and categorical features were selected (e.g. league, nationality, age, skill ratings).

The project consists of the following stages:

1. **Exploratory Data Analysis (EDA)**  
   - Visualizations of feature relationships (league, age, position, etc.)
   - Dimensionality reduction with PCA

2. **Machine Learning Models**  
   - Linear regression (custom implementation, scikit-learn, PyTorch)
   - `RandomForestRegressor`, `SVR`, `DecisionTreeRegressor`
   - MSE-based comparison of model accuracy

3. **Model Optimization**  
   - Cross-validation (KFold)
   - Polynomial feature expansion (degrees 1–10) and convergence analysis
   - L1/L2 regularization
   - Data balancing (SMOTE, undersampling)
   - Hyperparameter tuning using GridSearchCV

4. **Ensemble Methods**  
   - `VotingRegressor`, `StackingRegressor`, `Mixture of Experts`
   - **Best performance achieved by Mixture of Experts**

---

## 🛠️ Technologies Used

- Python 3.11
- `scikit-learn`
- `numpy`, `pandas`, `seaborn`, `matplotlib`
- `PyTorch`
- `imbalanced-learn` (SMOTE)
- Jupyter Notebook

---

## 📊 Results

- The **Mixture of Experts** model achieved the lowest MSE by dynamically weighting individual expert predictions per sample.
- Linear models performed adequately but struggled with capturing complex relationships.
- Tree-based models (RandomForest, DecisionTree) handled non-linearity effectively.
- Ensemble methods like Voting and Stacking did not outperform the best single model.

---

## 📁 Repository Structure

```
├── EDA/
│   ├── __init__.py
│   ├── data_analysis_main.py
│   └── reader.py
│   └── stats.py
├── advanced_ML_models.ipynb
├── simple_ML_models.ipynb
├── players_22.csv
├── requirements.txt
└── README.md
```
---

## 🔧 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/DamianAdamek/FIFA-22-EDA-ML.git
   cd FIFA-22-EDA-ML
   ```

2. Install dependencies (via `pip` or `conda`):
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the notebooks:
   ```bash
   jupyter notebook
   ```

---

## 📌 Final Notes

This project demonstrates how various regression techniques—including custom implementations—can be applied to real-world data. The use of optimization methods and ensemble learning provided deeper insight into model behavior, stability, and performance under different conditions.

---

## 📬 Contact

For questions or feedback:  
Damian Adamek – *damada393@gmail.com*