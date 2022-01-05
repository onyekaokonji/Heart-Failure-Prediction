import streamlit as st
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report


# Title of Web app
st.title("Dexter Technologies Image Classification Web App")

st.write("This is a web application for the detection of pneumonia using Convolutional Neural Networks for the classification of images")


# Centre Image
background_image = cv2.imread("heart-failure.jpeg")
image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (1280, 640))
st.image(image)
st.caption('what heart failure looks like')

dataset = st.sidebar.selectbox('Select dataset', ('Heart Failure', 'Other'))

def data_to_use(name):
	data = None
	if name == 'Heart Failure':
		data = pd.read_csv("heart_failure_clinical_records_dataset.csv")

	return data 

data = data_to_use(dataset)

st.dataframe(data)

st.write("Dataset size is :",data.shape)

st.text("---" * 30)

st.dataframe(data.head(11))

st.text("---" * 30)

# checking for missing values
missing_values = data.isnull().sum()

st.write("There are no missing values in columns")

st.write(missing_values)

st.text("---" * 30)


st.write("Histogram distribution of target values")

fig, ax = plt.subplots()
ax.hist(data['DEATH_EVENT'])

st.pyplot(fig)

X = data.iloc[:, 0:12]
Y = data.iloc[:, 12]

s_scaler = StandardScaler()
X = s_scaler.fit_transform(X)

x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size = 0.2)

classifier = st.sidebar.selectbox("Model", ("Logistic Regression", "Decision Tree Classifier"))

def model_to_use(model):
	models = None
	if model == "Logistic Regression":
		models = LogisticRegression(solver = 'liblinear', class_weight = 'balanced')
	else:
		models = DecisionTreeClassifier(random_state = 0)

	return models

models = model_to_use(classifier)

training = models.fit(x_train, y_train)

st.text("---" * 30)

st.write("Model Prediction")
st.write("Prediction on sample test set")

predictions = models.predict(x_val)

st.write(predictions)

# Plotting confusion matrix
c_matrix = confusion_matrix(y_val, predictions, labels = training.classes_)
c_matrix_display = ConfusionMatrixDisplay(confusion_matrix = c_matrix, display_labels=training.classes_)
c_matrix_display.plot()
plt.show()

st.pyplot()

