import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv(r'C:\Coding\Ai&ML\Placement Project\placement-prediction\college_student_placement_dataset.csv')

a = df.iloc[:, [1,3]]
b = df.iloc[:, 9]

cat_cols = a.select_dtypes(include=['object']).columns
for col in cat_cols:
    a.loc[:, col] = a[col].map({'No': 0, 'Yes': 1})

b = b.map({'No': 0, 'Yes': 1})


atrain, atest, btrain, btest = train_test_split(a, b, test_size=0.1)

scaler = StandardScaler()
atrain = scaler.fit_transform(atrain)
atest = scaler.transform(atest)

lr = LogisticRegression()
lr.fit(atrain, btrain)

prediction = lr.predict(atest)

print(accuracy_score(btest,prediction))

pickle.dump(lr,open('placement.pkl','wb'))