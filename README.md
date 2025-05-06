

#  Social Media Sentiment Analysis

This project performs sentiment analysis on tweets using the **Twitter Sentiment Analysis Dataset** from Kaggle. It applies NLP techniques for data cleaning and uses models like **Naive Bayes** and **TF-IDF with ML classifiers** to classify sentiments as **positive** or **negative**.

## 📁 Dataset

The dataset used is:
**`training.1600000.processed.noemoticon.csv`** from Kaggle's Twitter Sentiment Analysis dataset

* **Sentiment**: 0 = Negative, 4 = Positive (converted to 1 for binary classification)
* **Text**: Actual tweet text

👉 You can download it from Kaggle: [Twitter Sentiment Analysis Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)

---

## 🛠️ Technologies Used

* Python
* NLTK
* NumPy / Pandas
* Matplotlib
* Scikit-learn

---

## 🧹 Data Preprocessing

1. **Drop irrelevant columns**: Keep only `Sentiment` and `Text`.
2. **Sentiment Mapping**: 4 → 1 (positive)
3. **Downsampling**: Balance dataset to have equal positive and negative samples.
4. **Text Cleaning**:

   * Lowercasing
   * Remove stopwords & punctuation
   * Remove digits, tags, special characters
   * Lemmatization using `WordNetLemmatizer`

---

## 📊 Exploratory Data Analysis (EDA)

* Distribution of sentiments before and after balancing
* Sample visualization using histograms
* Top features contributing to sentiment classification

---

## 🔍 Model Building

### 1. NLTK Naive Bayes Classifier

* Token-based feature extraction
* Accuracy:

  * **Training**: \~86%
  * **Testing**: \~76%

### 2. TF-IDF + ML Classifiers

* Feature extraction using `TfidfVectorizer`
* Models:

  * **Multinomial Naive Bayes**
  * **Bernoulli Naive Bayes**
  * **Linear SVM**
* Model evaluation using accuracy and confusion matrix

---

## 📈 Results

| Model                   | Accuracy (Test) |
| ----------------------- | --------------- |
| NLTK Naive Bayes        | 76%             |
| Multinomial NB (TF-IDF) | \~84–86%        |
| Linear SVM (TF-IDF)     | \~88–90%        |

---

## 📌 Future Work

* Hyperparameter tuning using GridSearchCV
* Adding emojis/emoticons handling
* Applying deep learning models (e.g., LSTM, BERT)


---




## 🙌 Acknowledgements

* Kaggle Twitter Dataset by [Kaggle user kazanova](https://www.kaggle.com/kazanova/sentiment140)
* NLTK and Scikit-learn documentation

---

Would you like me to generate this as a downloadable `README.md` file for GitHub?

