#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv("E:\spam.csv", encoding = 'latin-1')
df


# In[3]:


df.drop(columns = {'Unnamed: 2','Unnamed: 3','Unnamed: 4'},axis=1,inplace = True)


# In[4]:


df


# In[5]:


df=df.rename(columns={'v1':'Category','v2':'Text'})


# In[6]:


df


# In[7]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[8]:


df['Category']=le.fit_transform(df['Category'])


# In[9]:


df.info()


# In[10]:


df['Category'].value_counts()


# In[11]:


df.isna().sum()


# In[12]:


df.duplicated().sum()


# In[13]:


df.drop_duplicates(inplace=True)


# In[14]:


df.duplicated().sum()


# In[15]:


df.shape


# ## EDA

# In[16]:


l=df['Category'].value_counts()


# In[17]:


plt.pie(l,labels=['ham','spam'],autopct='%0.2f')
plt.show()


# #### Big chunk of ham and very less of spam so out data is not balanced

# ## nltk is basically natural language tool kit 

# In[18]:


import nltk


# In[19]:


get_ipython().system('pip install nltk')


# In[20]:


nltk.download('punkt')


# In[21]:


# num of char
df['num_of_characters']=df['Text'].apply(len)


# In[22]:


df.head()


# In[23]:


#num of words
df['num_words']=df['Text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[24]:


df.head()


# In[25]:


# num of sentences
df['num_sentences']=df['Text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[26]:


df.head()


# In[27]:


df[['num_of_characters','num_words','num_sentences']].describe()


# In[28]:


# Targeting ham
df[df['Category']==0][['num_of_characters','num_words','num_sentences']].describe()


# In[29]:


# targeting spam
df[df['Category']==1][['num_of_characters','num_words','num_sentences']].describe()


#   ## Visualization

# In[30]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[31]:


plt.figure(figsize=(10,6))
sns.histplot(df[df['Category']==0]['num_of_characters'])
sns.histplot(df[df['Category']==1]['num_of_characters'],color='red')


# In[32]:


plt.figure(figsize=(10,6))
sns.histplot(df[df['Category']==0]['num_words'])
sns.histplot(df[df['Category']==1]['num_words'],color='red')


# In[33]:


plt.figure(figsize=(10,6))
sns.histplot(df[df['Category']==0]['num_sentences'])
sns.histplot(df[df['Category']==1]['num_sentences'],color='red')


# In[34]:


sns.pairplot(df,hue='Category')


# In[35]:


# to store for corelation
data=df[['Category','num_of_characters','num_words','num_sentences']]
data


# In[36]:


# to use data variable for corelation
sns.heatmap(data.corr(), annot= True)


# ## Data processing

# In[37]:


import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string


# In[38]:


nltk.download('stopwords')


# In[39]:


ps=PorterStemmer()


# In[40]:


def transform_text(Text):
    Text = Text.lower()
    Text = nltk.word_tokenize(Text)
    
    y=[]
    for i in Text:
        if i.isalnum():
            y.append(i)
            
            Text =y[:]
            y.clear()
            
    for i in Text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
                    
                    
    Text = y[:]
    y.clear()
            
    for i in Text:
        y.append(ps.stem(i))
                
    return " ".join(y)
    


# In[41]:


# for check above code
tex = transform_text("i am no going")
tex


# In[42]:


df['transformed_text']=df['Text'].apply(transform_text)


# In[43]:


df.head()


# In[44]:


get_ipython().system('pip install wordcloud')


# In[45]:


from wordcloud import WordCloud
wc=WordCloud(width = 500, height=500,min_font_size=10,background_color='white')


# In[46]:


spam_wc = wc.generate(df[df['Category']==1]['transformed_text'].str.cat(sep=' '))


# In[47]:


plt.figure(figsize=(15,6))
plt.imshow(spam_wc)


# In[48]:


ham_wc = wc.generate(df[df['Category']==0]['transformed_text'].str.cat(sep=' '))


# In[49]:


plt.figure(figsize=(15,6))
plt.imshow(ham_wc)


# In[50]:


spam_corpus=[]
for msg in df[df['Category']==1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[51]:


len(spam_corpus)


# In[52]:


from collections import Counter

# Assuming spam_corpus is your text data
counter = Counter(spam_corpus).most_common(30)
dat = pd.DataFrame(counter, columns=['Word', 'Count'])

# Plotting using seaborn
sns.barplot(x='Word', y='Count', data=dat)

# Rotate x-axis labels for better readability
plt.xticks(rotation=90)

# Show plot
plt.show()


# In[53]:


ham_corpus=[]
for msg in df[df['Category']==0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[54]:


len(ham_corpus)


# In[55]:


counter = Counter(ham_corpus).most_common(30)
dat = pd.DataFrame(counter, columns=['word','count'])

sns.barplot(x= 'word',y='count', data=dat)
plt.xticks(rotation='vertical')
plt.show()


# In[56]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf=TfidfVectorizer(max_features=3000)


# In[57]:


X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[58]:


X.shape


# In[59]:


y =df['Category'].values


# In[60]:


from sklearn.model_selection import train_test_split


# In[61]:


x_train, x_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=2)


# In[62]:


from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score


# In[63]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb= BernoulliNB()


# In[64]:


gnb.fit(x_train,y_train)
p_gnb=gnb.predict(x_test)
print(accuracy_score(y_test,p_gnb))
print(confusion_matrix(y_test,p_gnb))
print(precision_score(y_test,p_gnb))


# In[65]:


mnb.fit(x_train,y_train)
p_mnb=mnb.predict(x_test)
print(accuracy_score(y_test,p_mnb))
print(confusion_matrix(y_test,p_mnb))
print(precision_score(y_test,p_mnb))


# In[66]:


bnb.fit(x_train,y_train)
p_bnb=bnb.predict(x_test)
print(accuracy_score(y_test,p_bnb))
print(precision_score(y_test,p_bnb))
print(confusion_matrix(y_test,p_bnb))


# In[67]:


get_ipython().system('pip install xgboost')


# In[68]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


# In[69]:


lr = LogisticRegression(solver='liblinear',penalty='l1')
svc =SVC(kernel = 'sigmoid', gamma=1.0)
dtc = DecisionTreeClassifier(max_depth=5)
mnb = MultinomialNB()
knc = KNeighborsClassifier()
rf=RandomForestClassifier(n_estimators=50, random_state=2)
abc = AdaBoostClassifier(n_estimators=50, random_state=2)
bgc = BaggingClassifier(n_estimators=50, random_state=2)
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)
gbc = GradientBoostingClassifier(n_estimators=50, random_state=2)
xgb = XGBClassifier(n_estimators=50, random_state=2)


# In[70]:


clfs = { 
    'sv' : svc,
    'lr': lr,
    'dt':dtc,
    'mn':mnb,
    'kn':knc,
    'rf':rf,
    'ab':abc,
    'bg':bgc,
    'et':etc,
    'gb':gbc,
    'xg':xgb
    
    }


# In[71]:


def train_classifier(clf,x_train,y_train,x_test,y_test):
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred,average='binary')
    
    return accuracy,precision


# In[72]:


train_classifier(rf,x_train,y_train,x_test,y_test)


# In[73]:


Accuracy_score=[]
Precision_score=[]

for name,clf in clfs.items():
    
    current_accuracy, current_precision = train_classifier(clf,x_train,y_train,x_test,y_test)
    
    print('For  - ',name)
    print('Accuracy  - ',current_accuracy)
    print('Precision  - ',current_precision)
    
    Accuracy_score.append(current_accuracy)
    Precision_score.append(current_precision)


# In[74]:


name


# In[80]:


performance_df = pd.DataFrame({'Algorithms':clfs.keys(),'Accuracy':Accuracy_score,'Precision':Precision_score}).sort_values('Precision',ascending=False)


# In[81]:


performance_df


# In[82]:


performance_df1 = pd.melt(performance_df,id_vars='Algorithms')
performance_df1


# In[85]:


sns.catplot(x='Algorithms',y='value',hue='variable',data=performance_df1, kind='bar',height=5)
plt.ylim(0.5,1.0)
plt.xticks(rotation=90)
plt.show()


# In[86]:


temp_df = pd.DataFrame({'Algorithms':clfs.keys(),'Accuracy_max_ft_3000':Accuracy_score,'Precision_max_ft_3000':Precision_score}).sort_values('Precision_max_ft_3000',ascending=False)


# In[87]:


new_df = performance_df.merge(temp_df,on='Algorithms')


# In[88]:


new_df_scale = new_df.merge(temp_df,on='Algorithms')


# In[89]:


temp_df = pd.DataFrame({'Algorithms':clfs.keys(),'Accuracy_num_char':Accuracy_score,'Precision_num_char':Precision_score}).sort_values('Precision_num_char',ascending=False)


# In[92]:


new_df_scale.merge(temp_df,on='Algorithms')


# In[98]:


# voting classifier

svc =SVC(kernel = 'sigmoid', gamma=1.0,probability=True)
mnb = MultinomialNB()
etc = ExtraTreesClassifier(n_estimators=50, random_state=2)

from sklearn.ensemble import VotingClassifier


# In[99]:


voting = VotingClassifier(estimators=[('sv',svc),('nb',mnb),('et',etc)],voting='soft')


# In[100]:


voting.fit(x_train,y_train)


# In[102]:


from sklearn.metrics import accuracy_score, precision_score
pred_v =voting.predict(x_test)
print('Accuracy',accuracy_score(y_test,pred_v))
print('Precision', precision_score(y_test,pred_v))


# In[103]:


# Applying stacking
estimators=[('sv',svc),('nb',mnb),('et',etc)]
final_estimator=RandomForestClassifier()


# In[104]:


from sklearn.ensemble import StackingClassifier


# In[105]:


clf = StackingClassifier(estimators=estimators, final_estimator=final_estimator)


# In[106]:


clf.fit(x_train,y_train)


# In[107]:


pred_v =clf.predict(x_test)
print('Accuracy',accuracy_score(y_test,pred_v))
print('Precision', precision_score(y_test,pred_v))


# ## Build Predict Model

# In[115]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


# In[117]:


# Example training data
train_texts = ["I love machine learning", "Python is great for data science", "Text classification is interesting"]
train_labels = [1, 1, 0]  # Example labels

# Set up feature extraction and model
feature_extraction = TfidfVectorizer()
lr = LogisticRegression()

# Create a pipeline that combines both steps
model = make_pipeline(feature_extraction, lr)

# Train the model
model.fit(train_texts, train_labels)


# In[120]:


# New input data
input_mail = ['congratulation,it is free']

# Predict using the trained model
prediction = model.predict(input_mail)
print(prediction)

if (prediction[0]==1):
    print('Ham Mail')
else:
    print('Spam Mail')


# In[ ]:




