# Machine Learning LA-I Assessment Study Guide

Since you have to write code **on paper** without any execution environment, it is critical to memorize the essential workflow, syntax, and function names for each experiment. This guide distills the core components of Experiments 1–4 to help you prepare for both the written code and the Viva.

---

## 🔬 Experiment 1: Data Preprocessing & EDA (Titanic)

### Core Concepts for Viva:
- **Why do we handle missing values?** Missing data can cause models to crash or introduce bias. Median is often used for numerical columns to avoid the impact of outliers.
- **Why scale data?** Standard scaling ($\mu=0, \sigma=1$) helps algorithms that compute distances (like KNN or SVM) or optimize via gradient descent converge faster and treat all features equally.

### Important Syntax to Memorize:
```python
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# 1. Loading and Inspecting
df = pd.read_csv("titanic.csv")
df.isnull().sum() # Check missing values
df.describe()     # Summary statistics

# 2. Handling Missing Data
df["Age"].fillna(df["Age"].median(), inplace=True)
df.drop(["PassengerId", "Cabin"], axis=1, inplace=True)

# 3. Categorical Encoding (Manual)
df["Sex"] = df["Sex"].replace("male", 0)
df["Sex"] = df["Sex"].replace("female", 1)

# 4. Feature Scaling
sc = StandardScaler()
# Note: StandardScaler requires a 2D array/dataframe
x[["Age", "Fare"]] = sc.fit_transform(x[["Age", "Fare"]])

# 5. Visualizations
sns.histplot(df["Age"], kde=True)
sns.boxplot(x=df["Fare"])
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='magma')
```

---

## 🔬 Experiment 2: Linear & Multiple Regression (USA Housing)

### Core Concepts for Viva:
- **What is Regression?** Predicting a continuous numerical value (e.g., house price).
- **Difference between Ridge (L2) and Lasso (L1)?** 
  - **Ridge** shrinks coefficients to prevent overfitting but doesn't make them exactly zero.
  - **Lasso** shrinks some coefficients to exactly zero, performing feature selection (handles outliers better).
- **Evaluation Metrics:** R² score indicates the goodness of fit (closer to 1 is better). RMSE interprets error in the original target's unit.

### Important Syntax to Memorize:
```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 1. Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Multiple Linear Regression
mlr = LinearRegression()
mlr.fit(X_train, y_train)
y_pred = mlr.predict(X_test)

# 3. Metrics (Learn to write these out!)
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2:", r2_score(y_test, y_pred))

# 4. Hyperparameter Tuning (Ridge/Lasso)
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_ridge = GridSearchCV(estimator=Ridge(), param_grid=param_grid, scoring='r2', cv=5)
grid_ridge.fit(X_train, y_train)

# Best Model
best_ridge = grid_ridge.best_estimator_
print(grid_ridge.best_params_)
```

---

## 🔬 Experiment 3: Concept Learning (Find-S & Candidate Elimination)

*Note: You implemented these from scratch. For paper tests, memorizing the logic loop is critical.*

### Core Concepts for Viva:
- **Find-S:** Starts with the most specific hypothesis (`[Null, Null,...]`) and only looks at **Positive ("Yes")** examples to generalize the hypothesis by replacing mismatches with `?`.
- **Candidate Elimination:** Maintains two boundaries: Specific (`S`) and General (`G`). It updates `S` for positive examples and `G` for negative examples, converging towards the version space.

### Important Logic/Syntax to Memorize (Find-S):
```python
def find_s(X, y):
    # Initialize hypothesis with the first positive example
    hypothesis = None
    
    for i in range(len(X)):
        if y[i] == "Yes":
            if hypothesis is None:
                hypothesis = X[i].copy()
            else:
                # Compare current example with hypothesis
                for j in range(len(hypothesis)):
                    if hypothesis[j] != X[i][j]:
                        hypothesis[j] = "?"
    return hypothesis
```

---

## 🔬 Experiment 4: Decision Trees & Classification (Wine Quality)

### Core Concepts for Viva:
- **Decision Trees:** Split data based on criteria like Entropy or Gini Impurity to maximize Information Gain.
- **Confusion Matrix:** Evaluates classification. Know what True Positives (TP), TN, FP, FN mean.
- **ROC-AUC:** ROC plots True Positive Rate vs False Positive Rate. AUC is the area under it (closer to 1.0 is better).

### Important Syntax to Memorize:
```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Converting Continuous target to Categorical (Binning)
df['quality_group'] = pd.cut(df['quality'], bins=[0,5,6,10], labels=['Low','Medium','High'])

# 2. Decision Tree Classifier
dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_model.fit(X_train, y_train)
y_pred = dt_model.predict(X_test)

# 3. Evaluation
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 4. Grid Search for Decision Trees
param_grid = {
    'criterion': ['entropy'],
    'max_depth': [3, 5, 7, 10, None],
    'min_samples_split': [2, 5, 10, 20]
}
grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)
```

## 📝 Tips for Writing Code on Paper:
1. **Imports:** Always write the necessary library imports at the top (e.g., `import pandas as pd`, `from sklearn.model_selection import train_test_split`). Even if you forget the exact module path, write the import intent.
2. **Indentation:** Make sure your indentations inside `for` loops and `if` statements are visually clear.
3. **Variables:** Be consistent with variables like `X_train`, `X_test`, `y_train`, `y_test`.
4. **Comments:** Write `# comments` before major steps. It shows the examiner you know *why* you are doing something, which helps cover up minor syntax mistakes.
