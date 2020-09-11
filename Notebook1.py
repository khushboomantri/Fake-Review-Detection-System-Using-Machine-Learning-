#!/usr/bin/env python
# coding: utf-8

# # Fake Spam Detection
# 

# 
# 
# Basically I am importing the datasets and storing the datas in three columns : 
# *Polarity of the review
# *Review itself
# *True or Deceptive as ('t' or 'd')
# 
# Then I am converting 't' to 1 and 'd' to 0 because I will be using this as my target value and the review as my feature.
# Then I am splitting the Review data into testing data and training data (0.3 and 0.7 respectively).
# Then I am using CountVectorizer() to extract numeric features of each of the review as classifier can only use numeric data to compute something.
# Then I am using MultinomialNB method classifier to classify the reviews as Deceptive/True.

# Dependencies I used :
# * os for loading os folder paths
# * pandas for making dataframes
# * numpy for making arrays
# * sklearn.metrics for accuracy score, precision score, recall score, f1 score
# * sklearn.cross_validation for splitting the dataset
# * CountVectorizer() for extracting features from text in numerical form
# * MultinomialNB for importing naive bayes multinomial method classifier

# **Importing all the dependencies that will be needed.**
# 

# In[1]:


import os
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from wordcloud import WordCloud
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# **Setting up the folder paths in which the dataset is presetn**
# 

# In[2]:


neg_deceptive_folder_path = './negative_polarity/deceptive_from_MTurk/'
neg_true_folder_path = './negative_polarity/truthful_from_Web/'
pos_deceptive_folder_path = './positive_polarity/deceptive_from_MTurk/'
pos_true_folder_path = './positive_polarity/truthful_from_TripAdvisor/'


# 
# **Initialising the lists in which the polarity, review and either it's fake or true will be stored**

# In[3]:


polarity_class = []
reviews = []
spamity_class =[]


# ** Since we have 5 folders in each folder in our dataset, I am using a for loop to iterate through each of the folder and collect datas (i.e Polarity, Review, Fake or True) and store**

# In[4]:


for i in range(1,6):
    insideptru = pos_true_folder_path + 'fold' + str(i) 
    insidepdec = pos_deceptive_folder_path + 'fold' + str(i)
    insidentru = neg_true_folder_path + 'fold' + str(i) 
    insidendec = neg_deceptive_folder_path + 'fold' + str(i) 
    pos_list = []
    for data_file in sorted(os.listdir(insidendec)):
        polarity_class.append('negative')
        spamity_class.append(str(data_file.split('_')[0]))
        with open(os.path.join(insidendec, data_file)) as f:
                contents = f.read()
                reviews.append(contents)
    for data_file in sorted(os.listdir(insidentru)):
        polarity_class.append('negative')
        spamity_class.append(str(data_file.split('_')[0]))
        with open(os.path.join(insidentru, data_file)) as f:
                contents = f.read()
                reviews.append(contents)
    for data_file in sorted(os.listdir(insidepdec)):
        polarity_class.append('positive')
        spamity_class.append(str(data_file.split('_')[0]))
        with open(os.path.join(insidepdec, data_file)) as f:
                contents = f.read()
                reviews.append(contents)
    for data_file in sorted(os.listdir(insideptru)):
        polarity_class.append('positive')
        spamity_class.append(str(data_file.split('_')[0]))
        with open(os.path.join(insideptru, data_file)) as f:
                contents = f.read()
                reviews.append(contents)


# ** Making the dataframe using pandas to store polarity, reviews and true or fake **

# In[5]:

data_fm = pd.DataFrame({'polarity_class':polarity_class,'review':reviews,'spamity_class':spamity_class})

data_fm.loc[data_fm['spamity_class']=='d','spamity_class']="__label__deceptive"
data_fm.loc[data_fm['spamity_class']=='t','spamity_class']="__label__true"
data_fm.sample(frac=1)

# ** Splitting the dataset to training and testing (0.7 and 0.3)**

# In[6]:

data_fm = pd.DataFrame({'polarity_class':polarity_class,'review':reviews,'spamity_class':spamity_class})
data_fm.loc[data_fm['spamity_class']=='d','spamity_class']=0
data_fm.loc[data_fm['spamity_class']=='t','spamity_class']=1
data_x = data_fm['review']

data_y = np.asarray(data_fm['spamity_class'],dtype=int)

data_x


# In[ ]:

data_y

# ** Splitting the dataset to training and testing (0.8 and 0.2)**
X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2,random_state=35)
#training set
X_train
#test set
X_test

y_train

y_test

# ** Using CountVectorizer() method to extract features from the text reviews and convert it to numeric data, which can be used to train the classifier **

# *Using fit_transform() for X_train and only using transform() for X_test*

cv =  CountVectorizer()
X_traincv = cv.fit_transform(X_train)
X_testcv = cv.transform(X_test)

X_traincv

X_testcv

vectorizer_test = CountVectorizer(vocabulary=cv.vocabulary_)

cv.vocabulary_

# **Using Naive Bayes Multinomial method as the classifier and training the data**
nbayes = MultinomialNB()
nbayes.fit(X_traincv, y_train)


# **Predicting the fake or deceptive reviews**
# *using X_testcv : which is vectorized such that the dimensions are matched*
y_predictions = nbayes.predict(X_testcv)
y_result = list(y_predictions)
yp=["True" if a==1 else "Deceptive" for a in y_result]
X_testlist = list(X_test)
output_fm = pd.DataFrame({'Review':X_testlist ,'True(1)/Deceptive(0)':yp})

y_predictions

output_fm

print("Accuracy % :",metrics.accuracy_score(y_test, y_predictions)*100)

print("Recall Score: ",recall_score(y_test, y_predictions, average='micro') )

cnf_matrix = confusion_matrix(y_test, y_predictions)
cnf_matrix

print (confusion_matrix(y_test, y_predictions))

print (classification_report(y_test, y_predictions))

# In[31]:


#Checking the revies from Trip Advisor and Web
def test_string(s):
    X_testcv = (cv.transform([s]).toarray())
    y_predict = nbayes.predict(X_testcv)
    return y_predict


# In[32]:


test_string("The hotel was bad.The room had a 27-inch Samsung led tv, a microwave.The room had a double bed")


# In[33]:


test_string("My family and I are huge fans of this place. The staff is super nice, and the food is great. The chicken is very good, and the garlic sauce is perfect. Ice cream topped with fruit is delicious too. Highly recommended!")


# In[34]:


#MTurk
test_string("Truly, DO NOT stay at this hotel. When we arrived in our room, it was clear that the carpet hadn't been vacuumed")


# In[35]:



#Trip Advisor
test_string("The food is Asian fusion, and truly wonderful -- as a gourmand and life-long foodie, I appreciate great food when I taste it")


# In[36]:


#web
test_string("The elevator system was impossible. It seems they were trying to improve it but only made it worse.")


# In[37]:


f = sns.countplot(x='spamity_class', data=data_fm)
f.set_title("Sentiment distribution")
f.set_xticklabels(['Negative', 'Positive'])
plt.xlabel("");


# In[38]:


text = " ".join(review for review in data_fm.review)
print ("There are {} words in the combination of all review.".format(len(text)))


# In[39]:


wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white", stopwords=stopwords.words("english")).generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.margins(x=0, y=0)
plt.show();


# In[40]:


class_names = ["negative", "positive"]
fig,ax = plt.subplots(figsize=(6, 4))

sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="Blues", fmt="d",cbar=False, xticklabels=class_names, yticklabels=class_names)
plt.tight_layout()
plt.ylabel('Actual sentiment')
plt.xlabel('Predicted sentiment');
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')



