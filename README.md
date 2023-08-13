# Spam_classifier
A Spam Classifier project is made using Jupyter Notebook. 

### About
The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. 

### Dataset
The dataset contain one message per line. Each line is composed by two columns: v1 contains the label (ham or spam) and v2 contains the raw text.

### Python Libraries
1. **pandas:** pandas is an open-source data manipulation and analysis library for the Python programming language. It's often used to load, transform, and analyze structured data like CSV files or database tables.

2. **LabelEncoder:** LabelEncoder is a utility class provided by the scikit-learn library. It is used to convert categorical labels (text or numerical) into unique integers.

3. **nltk:** NLTK (Natural Language Toolkit) is a Python library for working with human language data, particularly in the field of natural language processing (NLP). It provides tools and resources for text processing, tokenization, stemming, tagging, parsing, and more.

4. **seaborn:** seaborn is a data visualization library built on top of matplotlib. It provides a high-level interface for creating attractive and informative statistical graphics. 

5. **stopwords:** Stopwords are commonly used words in a language (like "the," "and," "is," etc.) that are often filtered out during text processing or NLP tasks because they usually don't carry significant meaning or context.

6. **word_tokenize:** word_tokenize is a function from the NLTK library used for splitting text into individual words or tokens. It's a common preprocessing step in NLP.

7. **PorterStemmer:** PorterStemmer is a stemming algorithm provided by the NLTK library. Stemming is the process of reducing words to their base or root form, removing prefixes and suffixes. It's used to simplify words before text analysis.

8. **string:** In Python, the string module provides various constants and functions to work with strings, such as string formatting, manipulation, and handling.

9. **WordCloud:** WordCloud is a data visualization technique used to display the most frequent words in a textual dataset. It generates a visual representation where the size of each word corresponds to its frequency in the text.

10. **collections:** The collections module in Python provides specialized container datatypes beyond the built-in data structures like lists and dictionaries. It includes tools like Counter for counting occurrences of elements.

11. **Counter:** Counter is a class from the collections module that is used to count the occurrences of elements in a collection, often used to analyze the frequency of words in a text.

12. **CountVectorizer:** CountVectorizer is a class from the scikit-learn library used to convert a collection of text documents to a matrix of token counts. It's a common step in preparing text data for machine learning.

13. **TfidfVectorizer:** TfidfVectorizer is another class from scikit-learn used to convert a collection of text documents to a matrix of TF-IDF (Term Frequency-Inverse Document Frequency) features. It assigns weights to words based on their importance in a document and the entire corpus.

14. **GaussianNB, MultinomialNB, BernoulliNB:** These are classes from scikit-learn representing different variants of the Naive Bayes algorithm, a probabilistic classification technique.

15. **accuracy_score:** accuracy_score is a function from scikit-learn used to compute the accuracy of a classification model by comparing predicted labels to true labels.

16. **confusion_matrix:** confusion_matrix is a function from scikit-learn used to evaluate the performance of a classification model by showing the counts of true positive, true negative, false positive, and false negative predictions.

17. **precision_score:** precision_score is a function from scikit-learn used to calculate the precision of a classification model, which measures the ratio of correctly predicted positive observations to the total predicted positives.

18. **xgboost:** XGBoost is an open-source machine learning library that provides an efficient and scalable implementation of gradient boosting, a powerful ensemble learning technique.

19. **LogisticRegression:** LogisticRegression is a class from scikit-learn used for binary and multiclass logistic regression, a popular classification algorithm.

20. **DecisionTreeClassifier:** DecisionTreeClassifier is a class from scikit-learn used to create and train decision tree models for classification tasks.

21. **KNeighborsClassifier:** KNeighborsClassifier is a class from scikit-learn used for k-nearest neighbors classification, a simple and intuitive classification algorithm.

22. **RandomForestClassifier:** RandomForestClassifier is a class from scikit-learn used to create and train random forest models, which are ensembles of decision trees.

23. **AdaBoostClassifier:** AdaBoostClassifier is a class from scikit-learn used for AdaBoost (Adaptive Boosting) classification, another ensemble learning technique.

24. **BaggingClassifier:** BaggingClassifier is a class from scikit-learn used for bagging-based ensemble classification, where multiple base models are trained on subsets of the training data.

25. **ExtraTreesClassifier:** ExtraTreesClassifier is a class from scikit-learn used to create and train extremely randomized trees, which are similar to random forests but with additional randomness.

26. **GradientBoostingClassifier:** GradientBoostingClassifier is a class from scikit-learn used for gradient boosting, a machine learning technique that builds a model in a stage-wise manner.

27. **XGBClassifier:** XGBClassifier is a class from the XGBoost library specifically designed for classification tasks.

28. **StackingClassifier:** StackingClassifier is not a specific library, but rather a concept. It refers to a machine learning technique where multiple models (classifiers) are trained and their predictions are combined using another model (meta-classifier) to make the final prediction.
