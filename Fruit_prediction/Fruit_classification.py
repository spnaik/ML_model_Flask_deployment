import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Import data
data = pd.read_csv('fruits.csv')
data.head()

fruit_dict = {1:'apple',2:'mandarin',3:'orange',4:'lemon'}

# Features
features = ['mass', 'width', 'height', 'color_score']
X = data.iloc[:,3:]
y = data.iloc[:,0]

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X)

# Test-train split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Model training

rf = RandomForestClassifier(n_estimators=10)
rf.fit(X_train, y_train)

pickle.dump(rf, open('model.pkl','wb'))

print('Score on test: {}'.format(rf.score(X_test, y_test)))

model = pickle.load(open('model.pkl','rb'))
print(fruit_dict[model.predict([[1, 2, 3,4]])[0]])