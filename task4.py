# Spam detection with Naive Bayes import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer from sklearn.naive_bayes import MultinomialNB from sklearn.model_selection import train_test_split from sklearn.metrics import accuracy_score 
 
# Load sample spam dataset 
df = pd.read_csv("https://raw.githubusercontent.com/justmarkham /pycon-2016-tutorial/master/data/sms.tsv", sep="\t", names=["label", "message"]) 
 
vectorizer = CountVectorizer() 
X = vectorizer.fit_transform(df['message']) y = df['label'] 

 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 
 
model = MultinomialNB() model.fit(X_train, y_train) y_pred = model.predict(X_test) 
 
print("Accuracy:", accuracy_score(y_test, y_pred)) 

