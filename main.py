#URL : https://www.kaggle.com/datasets/colormap/spambase

import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model  import LogisticRegression 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score , confusion_matrix 
import matplotlib.pyplot as plt
import seaborn as sns 

# Load the dataset 

data = pd.read_csv('spambase.csv')
X =  data.drop('spam', axis=1)
y = data['spam']

# Split the data into training and testing sets 

X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(X_test)


# Evaluate the model 
accuracy  = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Confusion Matrix (visualization))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True , fmt='d')
plt.title('Confusion Matrix')
plt.show()





