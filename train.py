import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
data_dic = pickle.load(open('./data.pickle', 'rb'))

# Ensure all data samples have the same length (42 features)
data_list = data_dic['data']
desired_length = 42
uniform_data = [i[:desired_length] if len(i) > desired_length else i + [0]*(desired_length - len(i)) for i in data_list]
data = np.asarray(uniform_data)
labels = np.asarray(data_dic['labels'])

# Split the data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize and train the model
model = RandomForestClassifier()

# Cross-validation to verify model performance
cv_scores = cross_val_score(model, x_train, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Average cross-validation score: {np.mean(cv_scores)}")

# Fit the model
model.fit(x_train, y_train)

# Evaluate the model
y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f"Accuracy on test set: {score * 100}%")

# Save the model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

