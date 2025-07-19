# Fake-News-Detection-using-Python-
Fake News Detection uses Python, NLP, and ML to classify news as real or fake. It preprocesses text, applies TF-IDF, and trains models like Logistic Regression and Decision Tree. Visuals include WordCloud and confusion matrix. It’s deployable via Flask or Streamlit for real-time prediction
Here's a well-structured **project summary** in 200 words, based on your shared code and including sections for **Tools, Methodology, Features, and Code Summary**:

---

### 📌 **Fake News Detection Using Machine Learning**

**Domain:** Natural Language Processing (NLP) | **Language:** Python
**Dataset:** `News.csv` with `text` and `class` columns

---

### 🛠️ **Tools & Libraries Used:**

* **Python Libraries:** `pandas`, `numpy`, `seaborn`, `matplotlib`, `nltk`, `sklearn`
* **Modeling:** `LogisticRegression`, `DecisionTreeClassifier`
* **Text Vectorization:** `TfidfVectorizer`, `CountVectorizer`
* **Visualization:** `WordCloud`, `ConfusionMatrixDisplay`
* **Deployment:** `Flask` or `Streamlit` for front end

---

### 🔍 **Methodology:**

1. **Data Preprocessing**: Clean text by removing punctuation and stopwords using NLTK.
2. **Shuffling & Resetting Index**: Ensures randomness in training data.
3. **Vectorization**: Convert cleaned text into numerical features using TF-IDF.
4. **Model Training**: Apply ML models (Logistic Regression, Decision Tree).
5. **Evaluation**: Use `accuracy_score` and **confusion matrix** to measure performance.
6. **Visualization**: Display top frequent words and class balance using seaborn and bar plots.

---

### 🌐 **Features:**

* Interactive input for users to check news authenticity.
* WordCloud for real vs fake news.
* Bar chart showing most common keywords.

---

### 🔢 **Code Summary:**

* `data = pd.read_csv(...)`: Load dataset
* `preprocess_text()`: Clean text
* `TfidfVectorizer()`: Vectorize text
* `model.fit()`: Train model
* `accuracy_score()`: Evaluate
* `ConfusionMatrixDisplay`: Visualize results

---


