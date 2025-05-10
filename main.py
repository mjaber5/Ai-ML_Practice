import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# LabelEncoder: converts text labels into numbers (e.g., 'tcp' → 0).
# StandardScaler: standardizes feature values (mean=0, std=1).
# train_test_split: splits the dataset into training and testing sets.
# accuracy_score: evaluates how accurate the model is.
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# = Load Preprocessed CSV Data =
df = pd.read_csv('data_train.csv')

# = Encode Categorical Features =
# Specifies which columns contain categorical (non-numeric) values.
# Prepares a dictionary to store encoders for each column.
label_cols = ['protocol_type', 'service', 'flag']
label_encoders = {}

# Create a label encoder.
# Fit and transform the column to numerical values.
# Save the encoder for later use (optional).
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target labels (normal/attack types)
# Converts attack labels (e.g., 'normal', 'neptune') into numbers.
# This is necessary for classification models.
target_encoder = LabelEncoder()
df['class'] = target_encoder.fit_transform(df['class'])


# == Split Features and Labels ==
# X: input features (all columns except 'class').
# y: output label (just the 'class' column).
X = df.drop('class', axis=1)
y = df['class']

# == Normalize Features ==
# StandardScaler() normalizes the features:
# Mean = 0
# Standard deviation = 1
# Ensures features are scaled evenly for models like SVM and KNN.
scaler = StandardScaler()
X = scaler.fit_transform(X)

# === 5. Train-Test Split ===
# Splits the data:
# 80% for training
# 20% for testing
# random_state=42 ensures you get the same split every time (for reproducibility).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# == Data Exploration ==

# Class Distribution.
# Displays how many samples are in each class.
# Helps you understand class imbalance (e.g., too many attacks vs. normal samples).
plt.figure(figsize=(8, 4))
sns.countplot(x=y)
plt.title("Class Distribution (0 = normal, others = attack types)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# Feature Correlation Heatmap
# Visualizes how features relate to each other.
# Useful for identifying redundant or strongly correlated features.
plt.figure(figsize=(12, 6))
sns.heatmap(pd.DataFrame(X).corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# = Machine Learning Techniques =

### --- K-Nearest Neighbors (KNN) ---
# Uses the 5 nearest neighbors to classify each test point.
# Simple and effective for many problems.
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

### --- Support Vector Machine (SVM) ---
# Tries to find the best decision boundary between classes.
# Works well for high-dimensional data.
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

### --- Decision Tree (Gini) ---
# Builds a tree using the Gini Index to decide splits.
# Easy to understand and interpret.
dt_gini = DecisionTreeClassifier(criterion='gini')
dt_gini.fit(X_train, y_train)
y_pred_gini = dt_gini.predict(X_test)
print("Decision Tree (Gini) Accuracy:", accuracy_score(y_test, y_pred_gini))

### --- Decision Tree (Entropy) ---
# Similar to Gini, but uses entropy to measure impurity.
dt_entropy = DecisionTreeClassifier(criterion='entropy')
dt_entropy.fit(X_train, y_train)
y_pred_entropy = dt_entropy.predict(X_test)
print("Decision Tree (Entropy) Accuracy:", accuracy_score(y_test, y_pred_entropy))

### --- Neural Network (MLPClassifier) ---
# A small neural network with:
# 2 hidden layers (64 and 32 neurons)
# Trains for a maximum of 300 iterations
# Can model complex patterns in the data.
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
print("Neural Network (MLP) Accuracy:", accuracy_score(y_test, y_pred_mlp))

### --- Random Forest Classifier ---
# An ensemble of 100 decision trees.
# More robust and less overfitting than a single decision tree.
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))