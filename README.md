# 💬 NLPR Project – Part 1: Emotion Detection from Tweets

![Python](https://img.shields.io/badge/Made%20with-Python-blue.svg?logo=python)
![Scikit-learn](https://img.shields.io/badge/Library-Scikit--learn-yellow.svg?logo=scikit-learn)
![Keras](https://img.shields.io/badge/Model-Keras%20%7C%20XGBoost%20%7C%20CatBoost%20%7C%20SGD%20-blue.svg)
![Status](https://img.shields.io/badge/Progress-Completed-brightgreen)

This project tackles multi-class emotion classification using tweets. The primary goal is to classify text into emotions like `anger`, `fear`, `joy`, and `sadness`. This pipeline includes text cleaning, data visualization, training over 20 ML/DL models, and choosing the best-performing model based on rigorous evaluation metrics.

---

## 📂 Dataset Overview

- **Source**: [Kaggle - Emotion Dataset](https://www.kaggle.com/datasets/parulpandey/emotion-dataset)
- **Classes Used**: `anger`, `fear`, `joy`, `sadness` (removed `love` due to class imbalance)
- **Use Case**: Mental health monitoring, sentiment detection in social media, crisis detection.

---

## 🔍 Exploratory Data Analysis (EDA)

- Cleaned text via stopword removal, punctuation stripping, lowercasing
- Tokenization using `nltk`
- Frequency plots of top words per class
- Text length, character count, and unique word distribution per emotion
- Word clouds for each label
- First/last word frequency analysis

---

## 🧪 Feature Engineering

- **TF-IDF vectorization** (`max_features=1000`) on preprocessed text
- **Label Encoding** for target emotion labels
- **Train/Test Split** (80/20)

---

## 🧠 Models Trained (Total: 21)

| Category           | Models                                                                 |
|--------------------|------------------------------------------------------------------------|
| Linear Models       | Logistic Regression, Ridge Classifier, SGD, Perceptron, Passive Aggressive |
| Naive Bayes         | MultinomialNB, GaussianNB, BernoulliNB                                 |
| Tree-Based          | Decision Tree, Random Forest, Extra Trees, Gradient Boosting           |
| Ensemble Boosting   | AdaBoost, Bagging Classifier, XGBoost, CatBoost, LightGBM              |
| SVM                 | SVC (RBF), Linear SVC                                                  |
| KNN                 | K-Nearest Neighbors                                                    |
| Neural Network      | Keras Sequential (3-layer MLP with dropout)                            |

---

## 📊 Evaluation Metrics

- **Accuracy** on test set
- **Classification report** (precision, recall, F1-score)
- **Training time** for all models
- **Confusion matrix** per model
- **ROC-AUC curves** (multi-class micro-averaged for SGD)
- **Learning curves**: training vs validation accuracy over dataset sizes
- **Cross-validation**: 7-fold CV for selected models

---

## 🔧 Hyperparameter Tuning

- Models tuned with `GridSearchCV`: SGD, Logistic Regression, Ridge Classifier, XGBoost
- Final tuned parameters are summarized
- Training times and accuracy tracked during tuning

---

## 📈 Final Model Selection: **SGD Classifier**

### ✅ Why SGD?
- **Accuracy**: Consistent ~92% with fast training (< 1s)
- **Efficiency**: Fastest among models with comparable performance
- **Simplicity**: Linear model with interpretable parameters
- **Scalability**: Handles large text data via online learning
- **Generalization**: Lower risk of overfitting with L1 regularization

---

## 📊 Summary Table (Refined Models)

| Model              | Accuracy | Training Time | Best Params |
|-------------------|----------|----------------|-------------|
| **SGD Classifier** | 0.92     | ~0.6s         | hinge, L1, α=0.0001 |
| Ridge Classifier   | 0.91     | ~0.08s        | α=1.0, solver='lsqr' |
| XGBoost            | 0.90     | ~3.8s         | n_estimators=100, learning_rate=0.2 |
| Logistic Regression| 0.91     | ~0.24s        | C=10, solver='liblinear' |

---

## 📊 Visualizations

- Bar charts for metric comparisons (accuracy, precision, recall, F1)
- Learning curves for top models
- ROC curves (multi-class)
- Word clouds per emotion
- Confusion matrix heatmaps
- Class-wise performance bars

---

## 🏁 Conclusions

The SGD Classifier offered the best trade-off in:
- High accuracy and generalization
- Minimal computational cost
- Interpretability and multi-class handling

> This makes it suitable for production or mobile inference in emotion-based NLP applications.

---

## 🧠 Use Case Scenarios

- Mental health sentiment monitoring
- Emotion-aware chatbots
- Social media trend analysis
- Customer feedback emotion analysis

---

## 📁 Repository Structure

```
├── project_part_1.ipynb          # Full notebook with preprocessing, training, tuning, evaluation
├── training.csv                  # Input dataset
```

---

## 👤 Author

**Moanish Ashok Kumar**  
Applied AI Student · NLP & ML Enthusiast  
🔗 [LinkedIn](https://www.linkedin.com/in/moanish-ashok-kumar-086978272/)


---

## 🌟 Inspiration

Inspired by the CARER research paper on contextualized affect representations and motivated to apply emotion detection toward responsible AI use in mental health and human-computer interaction.
