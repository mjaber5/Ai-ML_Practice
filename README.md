# 📊 KDDTest+ Network Intrusion Detection using Machine Learning

This project demonstrates how various machine learning algorithms can be applied to detect network intrusions using the **KDDTest+** dataset. The workflow includes data loading, preprocessing, feature engineering, model training, evaluation, and visualization. This work is intended as an academic project to showcase practical applications of supervised ML techniques in cybersecurity.

---

## 🔍 Features

- ✅ ARFF to CSV conversion using `scipy.io.arff`
- 🧹 Label encoding for categorical features (`protocol_type`, `service`, `flag`)
- 🧠 Classification of network connections into "normal" or different types of attacks
- ⚖️ Feature scaling using `StandardScaler`
- 📊 Visualization of class distribution and correlation matrix
- 🤖 Machine Learning models:
  - K-Nearest Neighbors (KNN)
  - Support Vector Machine (SVM)
  - Decision Tree (Gini & Entropy)
  - Multi-Layer Perceptron (MLP) Neural Network
  - Random Forest Classifier

---

## 📁 Dataset

- **Name**: KDDTest+
- **Source**: NSL-KDD Dataset  
- **Format**: `.arff` (converted to `.csv`)
- **Use Case**: Intrusion Detection in computer networks

> The [NSL-KDD dataset](https://www.unb.ca/cic/datasets/nsl.html) was created to improve upon the KDD Cup 1999 dataset by removing redundant records and providing a more balanced test set.

---

## 🧪 Machine Learning Models

| Model                     | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| **K-Nearest Neighbors**  | Classifies a sample based on the majority class among its `k` nearest neighbors. |
| **SVM**                  | Finds the optimal separating hyperplane between classes using kernel tricks. |
| **Decision Tree (Gini)** | Uses the Gini index to create decision boundaries by splitting nodes.       |
| **Decision Tree (Entropy)** | Uses information gain (entropy) to split nodes.                          |
| **MLPClassifier**        | A neural network with hidden layers used for multi-class classification.    |
| **Random Forest**        | An ensemble of decision trees improving performance and reducing overfitting.|

---

## 📈 Visualizations

- 📌 **Class Distribution**: Helps visualize the frequency of normal vs. attack types  
- 📌 **Correlation Heatmap**: Shows the relationships between numerical features

---

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/kdd-intrusion-detection.git
   cd kdd-intrusion-detection
2. **Install required libraries**
   ```bash
   pip install -r requirements.txt
3. **Run the main script**
   ```bash
   python main.py

## 📦 Requirements
  ```bash
  pandas
  numpy
  matplotlib
  seaborn
  scikit-learn
  scipy
