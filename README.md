# Titanic Dataset - Exploratory Data Analysis (EDA)

This document contains exploratory data analysis performed on the Titanic dataset using Python libraries such as Pandas, Seaborn, and Matplotlib. The goal is to understand the structure, relationships, and key patterns within the dataset.

## Table of Contents
1. Summary Statistics
2. Histograms and Boxplots
3. Correlation Matrix and Pairplot
4. Pattern and Trend Analysis
5. Feature-Level Inferences

## 1. Summary Statistics

Basic descriptive statistics for numerical features such as Age, Fare, SibSp, and Parch were computed to understand the distribution and spread of data.

```python
df.describe()
df.median(numeric_only=True)
```

## 2. Histograms and Boxplots

Visualizations were created to observe the distribution and detect outliers in numeric columns.

### Histograms

```python
numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']
df[numeric_cols].hist(figsize=(12, 8), bins=20, edgecolor='black')
plt.tight_layout()
plt.show()
```

### Boxplots

```python
plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_cols):
    plt.subplot(2, 2, i + 1)
    sns.boxplot(data=df, y=col, color='skyblue')
    plt.title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()
```

## 3. Correlation Matrix and Pairplot

To analyze feature relationships and detect multicollinearity, a correlation matrix and pairplot were used.

### Correlation Heatmap

```python
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
plt.title("Correlation Matrix of Numeric Features")
plt.show()
```

### Pairplot

```python
selected_features = ['Survived', 'Pclass', 'Age', 'Fare', 'SibSp', 'Parch']
sns.pairplot(df[selected_features], hue='Survived', diag_kind='kde', corner=True)
plt.suptitle("Pairplot of Selected Features by Survival", y=1.02)
plt.show()
```

## 4. Pattern and Trend Analysis

Visualizations were created to identify patterns and trends in survival based on gender, age, and other features.

### Survival Count

```python
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()
```

### Survival by Gender

```python
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Sex")
plt.show()
```

### Age Distribution by Survival

```python
sns.kdeplot(data=df[df['Survived'] == 1]['Age'].dropna(), label='Survived')
sns.kdeplot(data=df[df['Survived'] == 0]['Age'].dropna(), label='Did Not Survive')
plt.legend()
plt.title("Age Distribution by Survival")
plt.show()
```

## 5. Feature-Level Inferences

Based on the above analysis, the following inferences were made:

| Feature  | Inference |
|----------|-----------|
| Survived | Most passengers did not survive, indicating a class imbalance |
| Sex | Females had significantly higher survival rates than males |
| Age | Younger passengers, especially children, were more likely to survive |
| Fare | Higher fare often correlated with higher survival probability |
| Pclass | First-class passengers had better survival outcomes compared to lower classes |
