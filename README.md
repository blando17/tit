#  Titanic Survival Prediction

A machine learning classification project that predicts passenger survival on the Titanic using logistic regression. The dataset includes demographic and passenger-specific information such as age, gender, ticket class, and more.

---

##  Project Overview

The Titanic dataset is a well-known dataset used for binary classification. This project aims to:
- Clean and preprocess the Titanic dataset.
- Perform exploratory data analysis.
- Encode categorical data.
- Train a logistic regression model to predict survival.
- Evaluate model accuracy.

---

##  Technologies Used

| Tool/Library         | Purpose                             |
|----------------------|-------------------------------------|
| Python               | Programming Language                |
| Pandas & NumPy       | Data manipulation and processing    |
| Matplotlib & Seaborn | Data visualization                  |
| scikit-learn         | Model training, evaluation, splitting |

---

##  Dataset Information

- **Total Records**: 891
- **Target Variable**: `Survived` (0 = No, 1 = Yes)
- **Features Used**:
  - `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`

---

##  Data Preprocessing

- Dropped:
  - `Cabin` (too many missing values)
  - `PassengerId`, `Name`, `Ticket` (not useful for prediction)

- Filled missing values:
  - `Age` → Mean
  - `Embarked` → Mode

- Categorical Encoding:
  - `Sex`: {'male': 0, 'female': 1}
  - `Embarked`: {'S': 0, 'C': 1, 'Q': 2}

---

##  Exploratory Data Analysis

Visualizations included:
- Countplot of survivors
- Survival distribution by sex
- Survival distribution by passenger class

```python
sns.countplot(x='Survived', data=titanic_data)
sns.countplot(x='Sex', hue='Survived', data=titanic_data)
sns.countplot(x='Pclass', hue='Survived', data=titanic_data)
```

---

##  Feature & Target Split

```python
X = titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Survived'])
y = titanic_data['Survived']
```

Train-test split:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
```

---

##  Model: Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
```

---

##  Model Evaluation

- **Training Accuracy**: `~80.8%`
- **Test Accuracy**: `~78.2%`

Evaluation was done using:
```python
accuracy_score(y_train, model.predict(X_train))
accuracy_score(y_test, model.predict(X_test))
```

---

##  Conclusion

This project demonstrates how logistic regression can be effectively used to predict survival outcomes based on passenger data. The model performs reasonably well and can serve as a baseline for further improvements.

---


---
