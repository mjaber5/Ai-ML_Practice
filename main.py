import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# LabelEncoder: converts text labels into numbers (e.g., 'tcp' → 0).
# StandardScaler: standardizes feature values (mean=0, std=1).
# train_test_split: splits the dataset into training and testing sets.
# accuracy_score: evaluates how accurate the model is.
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier


# = Load Preprocessed CSV Data =
# 42 feature | column
# 22545 Row  
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
print("\nKNN Accuracy:", f'{accuracy_score(y_test, y_pred_knn):.4f}')
accuracy = accuracy_score(y_test, y_pred_knn)
print(f"Accuracy (Recognition Rate): {accuracy:.2f} (Calculated)")

precision = precision_score(y_test, y_pred_knn, average='weighted')
print(f"Precision: {precision:.2f} (Calculated)")

recall = recall_score(y_test, y_pred_knn, average='weighted')
print(f"Sensitivity (Recall, True Positive Rate): {recall:.2f} (Calculated)")

f1 = f1_score(y_test, y_pred_knn, average='weighted')
print(f"F₁-score (Harmonic Mean of Precision and Recall): {f1:.2f} (Calculated)\n")

### --- Support Vector Machine (SVM) ---
# Tries to find the best decision boundary between classes.
# Works well for high-dimensional data.
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("\nSVM Accuracy:", f'{accuracy_score(y_test, y_pred_svm):.4f}')

accuracy = accuracy_score(y_test, y_pred_svm)
print(f"Accuracy (Recognition Rate): {accuracy:.2f} (Calculated)")

precision = precision_score(y_test, y_pred_svm, average='weighted')
print(f"Precision: {precision:.2f} (Calculated)")

recall = recall_score(y_test, y_pred_svm, average='weighted')
print(f"Sensitivity (Recall, True Positive Rate): {recall:.2f} (Calculated)")

f1 = f1_score(y_test, y_pred_svm, average='weighted')
print(f"F₁-score (Harmonic Mean of Precision and Recall): {f1:.2f} (Calculated)\n")

### --- Decision Tree (Gini) ---
# Builds a tree using the Gini Index to decide splits.
# Easy to understand and interpret.
dt_gini = DecisionTreeClassifier(criterion='gini')
dt_gini.fit(X_train, y_train)
y_pred_gini = dt_gini.predict(X_test)
print("Decision Tree (Gini) Accuracy:", f'{accuracy_score(y_test, y_pred_gini):.4f}')

accuracy = accuracy_score(y_test, y_pred_gini)
print(f"Accuracy (Recognition Rate): {accuracy:.2f} (Calculated)")

precision = precision_score(y_test, y_pred_gini, average='weighted')
print(f"Precision: {precision:.2f} (Calculated)")

recall = recall_score(y_test, y_pred_gini, average='weighted')
print(f"Sensitivity (Recall, True Positive Rate): {recall:.2f} (Calculated)")

f1 = f1_score(y_test, y_pred_gini, average='weighted')
print(f"F₁-score (Harmonic Mean of Precision and Recall): {f1:.2f} (Calculated)\n")

### --- Decision Tree (Entropy) ---
# Similar to Gini, but uses entropy to measure impurity.
dt_entropy = DecisionTreeClassifier(criterion='entropy')
dt_entropy.fit(X_train, y_train)
y_pred_entropy = dt_entropy.predict(X_test)
print("Decision Tree (Entropy) Accuracy:", f'{accuracy_score(y_test, y_pred_entropy):.4f}')

accuracy = accuracy_score(y_test, y_pred_entropy)
print(f"Accuracy (Recognition Rate): {accuracy:.2f} (Calculated)")

precision = precision_score(y_test, y_pred_entropy, average='weighted')
print(f"Precision: {precision:.2f} (Calculated)")

recall = recall_score(y_test, y_pred_entropy, average='weighted')
print(f"Sensitivity (Recall, True Positive Rate): {recall:.2f} (Calculated)")

f1 = f1_score(y_test, y_pred_entropy, average='weighted')
print(f"F₁-score (Harmonic Mean of Precision and Recall): {f1:.2f} (Calculated)\n")

### --- Neural Network (MLPClassifier) ---
# A small neural network with:
# 2 hidden layers (64 and 32 neurons)
# Trains for a maximum of 300 iterations
# Can model complex patterns in the data.
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
print("Neural Network (MLP) Accuracy:", f'{accuracy_score(y_test, y_pred_mlp):.4f}')

accuracy = accuracy_score(y_test, y_pred_mlp)
print(f"Accuracy (Recognition Rate): {accuracy:.2f} (Calculated)")

precision = precision_score(y_test, y_pred_mlp, average='weighted')
print(f"Precision: {precision:.2f} (Calculated)")

recall = recall_score(y_test, y_pred_mlp, average='weighted')
print(f"Sensitivity (Recall, True Positive Rate): {recall:.2f} (Calculated)")

f1 = f1_score(y_test, y_pred_mlp, average='weighted')
print(f"F₁-score (Harmonic Mean of Precision and Recall): {f1:.2f} (Calculated)\n")

### --- Random Forest Classifier ---
# An ensemble of 100 decision trees.
# More robust and less overfitting than a single decision tree.
# Calculate and print the evaluation metrics
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", f'{accuracy_score(y_test, y_pred_rf):.4f}')

accuracy = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy (Recognition Rate): {accuracy:.2f} (Calculated)")

precision = precision_score(y_test, y_pred_rf, average='weighted')
print(f"Precision: {precision:.2f} (Calculated)")

recall = recall_score(y_test, y_pred_rf, average='weighted')
print(f"Sensitivity (Recall, True Positive Rate): {recall:.2f} (Calculated)")

f1 = f1_score(y_test, y_pred_rf, average='weighted')
print(f"F₁-score (Harmonic Mean of Precision and Recall): {f1:.2f} (Calculated)\n")


# Store metrics in a dictionary
metrics = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-score': []
}

# Helper function to evaluate models
def add_metrics(name, y_true, y_pred):
    metrics['Model'].append(name)
    metrics['Accuracy'].append(round(accuracy_score(y_true, y_pred), 4))
    metrics['Precision'].append(round(precision_score(y_true, y_pred, average='weighted'), 4))
    metrics['Recall'].append(round(recall_score(y_true, y_pred, average='weighted'), 4))
    metrics['F1-score'].append(round(f1_score(y_true, y_pred, average='weighted'), 4))

# Add all model results
add_metrics('KNN', y_test, y_pred_knn)
add_metrics('SVM', y_test, y_pred_svm)
add_metrics('DT-Gini', y_test, y_pred_gini)
add_metrics('DT-Entropy', y_test, y_pred_entropy)
add_metrics('MLP', y_test, y_pred_mlp)
add_metrics('Random Forest', y_test, y_pred_rf)

# Convert to DataFrame for easier plotting
metrics_df = pd.DataFrame(metrics)

# === Plotting grouped bar chart ===
plt.figure(figsize=(12, 6))
bar_width = 0.2
x = range(len(metrics_df))

# Plot each metric
plt.bar([i - 1.5 * bar_width for i in x], metrics_df['Accuracy'], width=bar_width, label='Accuracy')
plt.bar([i - 0.5 * bar_width for i in x], metrics_df['Precision'], width=bar_width, label='Precision')
plt.bar([i + 0.5 * bar_width for i in x], metrics_df['Recall'], width=bar_width, label='Recall')
plt.bar([i + 1.5 * bar_width for i in x], metrics_df['F1-score'], width=bar_width, label='F1-score')

# Finalize chart
plt.xticks(x, metrics_df['Model'])
plt.ylabel('Score')
plt.ylim(0, 1.05)
plt.title('Comparison of Model Performance Metrics')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
