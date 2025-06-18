# NLP-Text-Processing

## 🎬 NLP Text Processing using IMDB Dataset

This project focuses on **Natural Language Processing (NLP)** techniques applied to the **IMDB movie reviews dataset**. It includes data preprocessing, vectorization, sentiment classification, and model evaluation using traditional machine learning models.

---

## 📌 Project Objectives

- Clean and preprocess raw IMDB review text.
- Convert text to numerical representations (TF-IDF, Count Vectorizer).
- Train sentiment classification models using scikit-learn.
- Evaluate model performance with accuracy, confusion matrix, and other metrics.
- Explore the use of pipelines and parameter tuning.

---

## 🛠️ Tech Stack

| Task              | Tools & Libraries                     |
|-------------------|----------------------------------------|
| Text Processing   | `nltk`, `re`, `string`, `sklearn`     |
| Vectorization     | `CountVectorizer`, `TfidfVectorizer`  |
| Modeling          | `LogisticRegression`, `NaiveBayes`    |
| Evaluation        | `classification_report`, `confusion_matrix` |
| Data Handling     | `pandas`, `numpy`                     |
| Visualization     | `matplotlib`, `seaborn`               |
| Notebook          | `Jupyter Notebook`                    |

---

## 📂 Project Structure

├── data/
│ └── imdb.csv
├── notebooks/
│ └── NLP_IMDB_Processing.ipynb
├── models/
│ └── saved_models.pkl
├── requirements.txt
├── README.md
└── .gitignore


---

## 🧪 Features Implemented

- Text preprocessing (lowercasing, stopword removal, stemming)
- Vectorization: Bag of Words and TF-IDF
- Model training: Naive Bayes, Logistic Regression
- Evaluation: Accuracy, precision, recall, F1-score
- ML Pipeline integration using `Pipeline()` from scikit-learn

---

## 🚀 How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/emmanueljirehb/NLP-Text-Processing-using-IMDB-Dataset.git
   cd NLP-Text-Processing-using-IMDB-Dataset

2. **pip install -r requirements.txt**

3. **jupyter notebook notebooks/NLP_IMDB_Processing.ipynb**

## 📊 Sample Results

- **Best Accuracy:** ~90% (using **TF-IDF + Logistic Regression**)
- **Key Insight:** Proper text cleaning and feature extraction greatly enhance model performance.

---

## 📌 Future Enhancements

- Apply deep learning models (e.g., **LSTM**, **BERT**).
- Build a web-based text sentiment classifier using **Streamlit** or **Flask**.
- Include hyperparameter tuning using **GridSearchCV**.


# NLP Interview questions 
1. What is the purpose of natural language processing?
2. what are the differnt types of data
3. give examples of structured/semi structured/unstrcutred data
4. what is tokentization
5. what is sentence tokenization
6. what is word tokenization
7. how do we get root words
8. what is stemming
9. what is lemmatization
10. what is the difference between stemming and lemmatization
11. what is the purpose of countvectorizer, TFIDF vectorizer
12. what is the differnce between countvectorizer and TFIDF vectorizer
13. what is stopwords
14. what is POS tagging?
15. what is corpus
16. Implement TFIDF for IMDB sentiment with all other classification algorithms
17. implement stemming or work lemmatizing inplace sentence lemmatizing
18. compare the a. time taken, b. accuracies of diffrent algorithms and stemming/word leammatizing
