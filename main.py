import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
label_cols = ['protocol_type', 'service', 'flag']
label_encoders = {}

for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target labels (normal/attack types)
target_encoder = LabelEncoder()
df['class'] = target_encoder.fit_transform(df['class'])

# == Split Features and Labels ==
X = df.drop('class', axis=1)
y = df['class']

# == Normalize Features ==
scaler = StandardScaler()
X = scaler.fit_transform(X)

# === 5. Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# == Data Exploration ==

# Class Distribution
plt.figure(figsize=(8, 4))
sns.countplot(x=y)
plt.title("Class Distribution (0 = normal, others = attack types)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# Feature Correlation Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(pd.DataFrame(X).corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# = Machine Learning Techniques =

### --- K-Nearest Neighbors (KNN) ---
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("KNN Accuracy:", accuracy_score(y_test, y_pred_knn))

### --- Support Vector Machine (SVM) ---
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

### --- Decision Tree (Gini) ---
dt_gini = DecisionTreeClassifier(criterion='gini')
dt_gini.fit(X_train, y_train)
y_pred_gini = dt_gini.predict(X_test)
print("Decision Tree (Gini) Accuracy:", accuracy_score(y_test, y_pred_gini))

### --- Decision Tree (Entropy) ---
dt_entropy = DecisionTreeClassifier(criterion='entropy')
dt_entropy.fit(X_train, y_train)
y_pred_entropy = dt_entropy.predict(X_test)
print("Decision Tree (Entropy) Accuracy:", accuracy_score(y_test, y_pred_entropy))

### --- Neural Network (MLPClassifier) ---
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
print("Neural Network (MLP) Accuracy:", accuracy_score(y_test, y_pred_mlp))

### --- Random Forest Classifier ---
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))