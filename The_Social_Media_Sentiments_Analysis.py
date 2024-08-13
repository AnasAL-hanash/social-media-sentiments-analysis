import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
df = pd.read_csv(r'C:\Users\ATC\Downloads\archive (2)\sentimentdataset.csv')

# Data cleaning and initial exploration
df.drop(columns='Unnamed: 0.1', inplace=True)
df.rename(columns={'Unnamed: 0': 'Id'}, inplace=True)
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Day'] = df['Timestamp'].dt.day
df['Month'] = df['Timestamp'].dt.month
df['Year'] = df['Timestamp'].dt.year
df['Text'] = df['Text'].str.strip()
df['Sentiment'] = df['Sentiment'].str.strip()
df['User'] = df['User'].str.strip()
df['Platform'] = df['Platform'].str.strip()
df['Hashtags'] = df['Hashtags'].str.strip()
df['Country'] = df['Country'].str.strip()

# Visualizations
df['Sentiment'].value_counts().nlargest(10).plot(kind='bar')
plt.title('Top 10 Sentiments based on Text')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

df['Platform'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Percentages of Platforms')
plt.legend()
plt.show()

df['Country'].value_counts().nlargest(10).plot(kind='bar')
plt.title('Top 10 Countries')
plt.legend()
plt.show()

df['Hashtags'].value_counts().nlargest(10).plot(kind='bar')
plt.title('Top 10 Hashtags')
plt.xlabel('Hashtags')
plt.ylabel('Count')
plt.legend()
plt.show()

# Describe numerical data
df.describe()

# Analyzing numerical columns
numerical_columns = df[['Day', 'Month', 'Year', 'Likes', 'Retweets']]

for col in numerical_columns.columns:
    print(f"Minimum {col}: {df[col].min()} | Maximum {col}: {df[col].max()}")

# Example visualization outside of loop
top_likes_platform = df.groupby('Platform')['Likes'].sum().nlargest(10)
top_likes_platform.plot(kind='bar')
plt.title('Top Platforms by Total Likes')
plt.xlabel('Platform')
plt.ylabel('Total Likes')
plt.show()

# Example code to analyze hashtags retweeted on Facebook
Facebook = df[df['Platform'] == 'Facebook']
H_R_f = Facebook.groupby('Hashtags')['Retweets'].max().nlargest(10).sort_values(ascending=False)
H_R_f.plot(kind='bar')
plt.title('Top 10 hashtags retweeted in Facebook')
plt.xlabel('Hashtags')
plt.ylabel('count')
plt.show()

# Split the data for training models
X = df[['Day', 'Month', 'Year', 'Likes', 'Retweets']]
y = df['Sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
LRclassifier = LogisticRegression(solver='liblinear', max_iter=5000)
LRclassifier.fit(X_train, y_train)
y_pred = LRclassifier.predict(X_test)
LRAcc = accuracy_score(y_pred, y_test)
print('Logistic Regression accuracy is: {:.2f}%'.format(LRAcc*100))

# Confusion Matrix for Logistic Regression
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix - Logistic Regression')
plt.show()

# Save the Logistic Regression model
import joblib
model_filename = 'model_of_data.joblib'
joblib.dump(LRclassifier, model_filename)
print(f"Model saved as {model_filename}")

# Load and test the saved model
model_data = joblib.load(model_filename)
test = model_data.predict(X_test)
print(test)

# K Neighbors Classifier
KNclassifier = KNeighborsClassifier(n_neighbors=200)
KNclassifier.fit(X_train, y_train)
y_pred = KNclassifier.predict(X_test)
KNAcc = accuracy_score(y_pred, y_test)
print('K Neighbors accuracy is: {:.2f}%'.format(KNAcc*100))

# Plot accuracy over different K values
scoreListknn = []
for i in range(1, 50):
    KNclassifier = KNeighborsClassifier(n_neighbors=i)
    KNclassifier.fit(X_train, y_train)
    scoreListknn.append(KNclassifier.score(X_test, y_test))
plt.plot(range(1, 50), scoreListknn)
plt.xticks(np.arange(1, 50, 1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.show()
KNAccMax = max(scoreListknn)
print("KNN Acc Max {:.2f}%".format(KNAccMax*100))

# Support Vector Machine (SVM)
SVCclassifier = SVC(kernel='linear', max_iter=50)
SVCclassifier.fit(X_train, y_train)
y_pred = SVCclassifier.predict(X_test)
SVCAcc = accuracy_score(y_pred, y_test)
print('SVC accuracy is: {:.2f}%'.format(SVCAcc*100))

# Decision Tree
DTclassifier = DecisionTreeClassifier(max_leaf_nodes=5)
DTclassifier.fit(X_train, y_train)
y_pred = DTclassifier.predict(X_test)
DTAcc = accuracy_score(y_pred, y_test)
print('Decision Tree accuracy is: {:.2f}%'.format(DTAcc*100))

# Plot accuracy over different leaf nodes in Decision Tree
scoreListDT = []
for i in range(2, 50):
    DTclassifier = DecisionTreeClassifier(max_leaf_nodes=i)
    DTclassifier.fit(X_train, y_train)
    scoreListDT.append(DTclassifier.score(X_test, y_test))
plt.plot(range(2, 50), scoreListDT)
plt.xticks(np.arange(2, 50, 5))
plt.xlabel("Leaf")
plt.ylabel("Score")
plt.show()
DTAccMax = max(scoreListDT)
print("DT Acc Max {:.2f}%".format(DTAccMax*100))

# Random Forest
RFclassifier = RandomForestClassifier(max_leaf_nodes=5)
RFclassifier.fit(X_train, y_train)
y_pred = RFclassifier.predict(X_test)
RFAcc = accuracy_score(y_pred, y_test)
print('Random Forest accuracy is: {:.2f}%'.format(RFAcc*100))

# Plot accuracy over different leaf nodes in Random Forest
scoreListRF = []
for i in range(2, 50):
    RFclassifier = RandomForestClassifier(n_estimators=1000, random_state=1, max_leaf_nodes=i)
    RFclassifier.fit(X_train, y_train)
    scoreListRF.append(RFclassifier.score(X_test, y_test))
plt.plot(range(2, 50), scoreListRF)
plt.xticks(np.arange(2, 50, 5))
plt.xlabel("RF Value")
plt.ylabel("Score")
plt.show()
RFAccMax = max(scoreListRF)
print("RF Acc Max {:.2f}%".format(RFAccMax*100))

# Model Comparison
compare = pd.DataFrame({'Model': ['Logistic Regression', 'K Neighbors', 'K Neighbors Max', 'SVM', 'Decision Tree', 'Decision Tree Max', 'Random Forest', 'Random Forest Max'],
                        'Accuracy': [LRAcc*100, KNAcc*100, KNAccMax*100, SVCAcc*100, DTAcc*100, DTAccMax*100, RFAcc*100, RFAccMax*100]})
compare = compare.sort_values(by='Accuracy', ascending=False)
print(compare)