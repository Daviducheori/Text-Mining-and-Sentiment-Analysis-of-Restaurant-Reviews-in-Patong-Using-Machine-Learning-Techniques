# Text-Mining-and-Sentiment-Analysis-of-Restaurant-Reviews-in-Patong-Using-Machine-Learning-Techniques
### Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Implementation in Python](#implementation-in-python)
- [Result and Discussion](#result-and-discussion)
- [Conclusion](#conclusion)


### Project Overview
The subject of text mining is experiencing rapid growth in the area of text sentiment analysis, also known as emotional polarity computing. This project analyses the reviews of customers in thirty restaurants in Patong, using sentiment analysis and text mining approaches.

### Programming Language
- Python
  
### Dataset
This study’s data was retrieved from the online traveling reviews site TripAdvisor. This website has grown to include 212,000 hotels, more than 30,000 places, and 74,000 destinations since it was first launched in the year 2000. The dataset consists of 53,644 entries and 5 columns. It also has a total number of 25 locations in Thailand consisting of 537 hotels/restaurants.

- Importing the Dataset: 
The data needed to be loaded into pandas. To make it easier to read the datasets using pandas, The datasets was downloaded from the source and placed in the same directory as the Jupyter notebook file. The pandas’ read csv() method was used because the dataset was in csv format.

- Selecting the Top 30 Hotels/Restaurants with the Highest Reviews in Patong: 
For a more accurate research outcome, Patong was chosen. Patong was the location with the highest number of restaurants. In terms of restaurants with the highest reviews, the top 30 restaurants with the highest reviews in Patong were selected for the analysis. In total, out of 53,644 reviews present in the dataset, 3,379 reviews were chosen for this research.

### Data Preprocessing
For the ensuing text mining processing, the data was further treated in the data pre-processing stage to create a suitable format. In text mining and sentiment analysis, pre-processing is crucial.

#### Removing Extra White Spaces 
This was used to clean out the data by removing extra spaces within the text in the reviews. It removes unwanted spaces at the start or end of text, it also shortens multiple spaces.

#### Removing Specific Words from the Reviews
Specific words that are widely used in tourists’ reviews and appears too often but cannot contribute to a considerable interpretation of the results and also do not give much information were removed.

#### Tokenization, Stop word Removal and Stemming

- Tokenization:
Tokenization, a form of lexical analysis, which is applied to break down sentences into words or phrases. It utilises the RegexpTokenizer from NLTK. This converts the character strings into tokens by splitting it into words using a regular expression. The regular expression ensures that only alphanumeric characters are tokenized, this removes punctuation excluding apostrophes which we want to retain in words.

- Stop Word Removal:
It was important to exclude stop words after word segmentation. Stop words such as articles (“the”), conjunctions (“and”), and propositions (“with), may not convey any useful information. Since learning that "the" and "a" are the most prevalent words in your dataset doesn't reveal much about the data. These words are removed. NLP Python libraries like NLTK provides an in-built stop word list which was used to remove stop words from the tokenized text.

- Stemming:
Stemming is the process of condensing word variants to their root forms. Thereby reducing the vocabulary size, thereby sharpening the results obtained. Porter’s Stemmer algorithm was used. Stemming was performed on the tokenized text.

### Implementation in Python
#### Sentiment Analysis

#### Importing Libraries
The relevant Python libraries were imported, including NumPy and Pandas for data manipulation, Matplotlib, and Seaborn for visualisation. Re for python regular expression, NLTK which is the natural language toolkit. It provides a variety of text processing libraries for many text mining and natural language processing tasks, including tokenization, stemming, and sentiment analysis.

#### Score Generation
Python was used to implement the sentimental analysis. The Python NLP package called NLTK was used to do the sentiment analysis. The SentimentIntensityAnalyzer within NLTK was used to get the negative score, neutral score, and positive score of the text. It generated a rating from -1 to 1, with a negative number denoting unfavourable sentiment and a positive number denoting favourable sentiment. A list comprehension was used to create new column for each from dictionaries using polarity scores method.

#### Compound Score Classification
A function was defined which classes a -0.05 compound score or below as negative, a 0.05 compound score and above as positive and a neutral classification for scores between 0.05 and -0.05. The resulting output was inputted into a new column named Sentiment which was used as the target variable in creating a model.

#### Creating a Model
- Preparation of Bag of Words/Generating a frequency Matrix:
A collection of words that maintains its multiplicity while ignoring the syntax. It involves the conversion of documents into a fixed length vector based on term frequency. Scikit Learn's CountVectorizer was imported to generate a frequency matrix. The tokenized text was vectorized using this method which created a frequency matrix.

- Segregation of Training and Test Data:
Machine learning involves feeding an algorithm a given amount of data and training it to recognise patterns in the data. Another dataset must be provided to the algorithm once it has learned the pattern in order to determine its level of understanding. As shown below, the data was split into a training and test datasets.


![image](https://github.com/Daviducheori/Text-Mining-and-Sentiment-Analysis-of-Restaurant-Reviews-in-Patong-Using-Machine-Learning-Techniques/assets/76125377/b9d23165-9c9d-4d29-9149-c5dfdebca873)

#### Class Imbalance

Here from the above visualization, it is clearly visible that the dataset is completely imbalanced. There were far more positive entries than there were negative entries or neutral entries. To create a balanced dataset, SMOTE was used from the imblearn library.

![image](https://github.com/Daviducheori/Text-Mining-and-Sentiment-Analysis-of-Restaurant-Reviews-in-Patong-Using-Machine-Learning-Techniques/assets/76125377/ec9a5489-614d-45f3-9088-305f7729090c)


#### Classification
After dividing the dataset into its component parts, we teach the algorithm how to classify the data by feeding it training data. Scikit Learn's Random Forest and MultinomialNB (Multinomial Naïve Bayes) classifier were instantiated.

#### Evaluation of Classification
The efficacy of any classification technique is evaluated by presenting a trained model the test dataset. Through this process, new data classes are projected, showing how effectively the algorithm has learned about the dataset. The number of True Positive (TP), False Positive (FP), True Negative (TN),and False Negative (FN) values produced by the provided dataset is contained in the confusion matrix that was created.



![image](https://github.com/Daviducheori/Text-Mining-and-Sentiment-Analysis-of-Restaurant-Reviews-in-Patong-Using-Machine-Learning-Techniques/assets/76125377/61615e3d-b04d-4ce0-8969-12f39681d4d7)

![image](https://github.com/Daviducheori/Text-Mining-and-Sentiment-Analysis-of-Restaurant-Reviews-in-Patong-Using-Machine-Learning-Techniques/assets/76125377/991680cb-1dd7-446b-832a-3256de05a204)

#### Multinomial Naïve Bayes Results

![image](https://github.com/Daviducheori/Text-Mining-and-Sentiment-Analysis-of-Restaurant-Reviews-in-Patong-Using-Machine-Learning-Techniques/assets/76125377/cce80e95-ec62-44fe-92ad-b81a784640d9)

![image](https://github.com/Daviducheori/Text-Mining-and-Sentiment-Analysis-of-Restaurant-Reviews-in-Patong-Using-Machine-Learning-Techniques/assets/76125377/ec4e9f49-066f-4483-80a2-f858424d90e2)

#### Random Forest Results

#### Visualization
The word cloud package and bar plots have been used to show the frequency of words in the reviews and the sentiment scores.

![Visualisation of Compound Scores](https://github.com/Daviducheori/Text-Mining-and-Sentiment-Analysis-of-Restaurant-Reviews-in-Patong-Using-Machine-Learning-Techniques/assets/76125377/e6715480-3c66-4abf-bf47-039d008d864e)

Visualisation of Compound Scores

![Visualisation of Positive Scores](https://github.com/Daviducheori/Text-Mining-and-Sentiment-Analysis-of-Restaurant-Reviews-in-Patong-Using-Machine-Learning-Techniques/assets/76125377/2954596d-b060-4040-822d-c31b8b5dff2d)

Visualisation of Postive Scores

![Visualisation of Negative Scores](https://github.com/Daviducheori/Text-Mining-and-Sentiment-Analysis-of-Restaurant-Reviews-in-Patong-Using-Machine-Learning-Techniques/assets/76125377/6534f8bd-ee52-4f2a-ba10-b53f2d09c181)

Visualisation of Negative Scores

![image](https://github.com/Daviducheori/Text-Mining-and-Sentiment-Analysis-of-Restaurant-Reviews-in-Patong-Using-Machine-Learning-Techniques/assets/76125377/6b74320f-a0c3-434b-89c4-5085cd6ab718)

Descriptive Statistics of Sentiment Scores

![image](https://github.com/Daviducheori/Text-Mining-and-Sentiment-Analysis-of-Restaurant-Reviews-in-Patong-Using-Machine-Learning-Techniques/assets/76125377/20f59f87-70cc-48ca-bed1-2ae5f7ebd055)
Percentage of Negative Reviews

![image](https://github.com/Daviducheori/Text-Mining-and-Sentiment-Analysis-of-Restaurant-Reviews-in-Patong-Using-Machine-Learning-Techniques/assets/76125377/a99be2ca-0217-4b3f-849f-9291d4547be2)

Negative Word Cloud

![image](https://github.com/Daviducheori/Text-Mining-and-Sentiment-Analysis-of-Restaurant-Reviews-in-Patong-Using-Machine-Learning-Techniques/assets/76125377/ca091e29-c613-4247-9a20-78c9037ee557)

Positive Word Cloud

### Result and Discussion
Examining the descriptive statistics of the sentiment scores of the reviews in the dataset shows that the results clearly reflect a majority of positive feedback. The median compound score is 0.83 – which means that over 50% of the reviews have a compound score greater than 0.83, which suggests strong positive sentiment. This was due to the fact that the collection of positive review terms had a bigger number and stronger discriminatory power than the collection of negative review words.

The graphical representation of the percentage of negative reviews showed Patong Seafood as the hotel with the highest percentage of negative reviews and Sam’s Steak and Gril as the restaurant with the lowest percentage of negative reviews.

In order to analyse the sentiment of Patong Seafood, which was the restaurant with the highest percentage of negative reviews, a positive and negative word cloud was generated. It may be possible to gain insight into a certain issue by analysing the frequency of appearance or merely the incremental count of appearance of specific words or phrases.

In the negative word cloud, we can see words such as “price”, “waiter”, “rude”, “hour”. In the positive word cloud, we can observe words such as “time”, “right”, “recommend”, “quality”. The Proposed model was tested with random forest and multinomial naïve bayes algorithms. Multinomial Naïve Bayes had a higher accuracy of 91% compared to 81% with Random Forest. Although both algorithms have a F1 score for the Neutral category.

### Conclusion
Sentiment analysis and text mining research is crucial in the modern day. The volume of data generated by social media is equally enormous, necessitating its analysis and interpretation. In the suggested methodology, machine learning techniques have been combined with a dictionary-based approach, also known as a lexicon-based approach. Each review underwent sentiment analysis, and the results were subsequently categorised using ML algorithms. i.e., Naïve Bayes and Random Forest The accuracy metrics for the dataset's Naive Bayes and Random Forest classifiers are shown above. In the same dataset, the accuracy of the Naive Bayes classifier for restaurant reviews was 91% while that of the Random Forest was 81.50%.

