# Placement Prediction Model
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
df = pd.read_csv("/content/placement.csv")
print(df.columns)
df = df.iloc[:, 1:]
plt.scatter(df['cgpa'], df['iq'], c=df['placement'])
plt.xlabel('CGPA')
plt.ylabel('IQ')
plt.title('Scatter Plot of CGPA vs IQ')
plt.show()
X = df.iloc[:, 0:2] 
y = df.iloc[:, -1] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
A = accuracy_score(y_test, y_pred)
print(f"Accuracy: {A}")
plot_decision_regions(X=X.values, y=y.values, clf=clf, legend=2)
plt.show()
