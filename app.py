import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine, load_breast_cancer, load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def return_data(dataset):
  if dataset == 'Wine':
    data = load_wine()
  elif dataset == 'Iris':
    data = load_iris()
  else:
    data = load_breast_cancer()
  
  df = pd.DataFrame(data.data, columns= data.feature_names, index = None)
  df['Class'] = data.target
  X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size = 0.2,random_state = 42)

  return X_train, X_test, y_train, y_test, df, data.target_names

##now create function to return the model
def getClassifier(classifier):
  if classifier == 'SVM':
    c = st.sidebar.slider(label= "Choose valud of C", min_value=0.0001, max_value=10.0)
    model = SVC(C = c)

  elif classifier == 'KNN':
    n = st.sidebar.slider(label = "Choose Number of Neighbors", min_value = 1, max_value = 20)
    model = KNeighborsClassifier(n_neighbors= n)

  else:
    max_depth = st.sidebar.slider("max_depth", 2, 10)
    n_estimator = st.sidebar.slider(label = "Choose number of neighbors", min_value = 1, max_value= 20)
    model = RandomForestClassifier(n_estimators= n_estimator, max_depth= max_depth, random_state = 42)
  return model

def getPCA(df):
  pca = PCA(n_components= 3)
  result= pca.fit_transform(df.loc[: , df.columns != 'Class'])
  df['pca_1'] = result[:, 0]
  df['pca_2'] = result[:, 1]
  df['pca_3'] = result[:, 2]
  return df


##Titile 
st.title("Classifier in Action")

##Description
st.text("Choose a dataset and a classifier in the sidebar. Input your values and get a prediction.")

#Sidebar
sidebar = st.sidebar
dataset = sidebar.selectbox("Which Dataset do you want to use ?", ("Wine", 'Breast Cancer', 'Iris'))
classifier = sidebar.selectbox("Which Classifier do you want to use ?", ("SVM", "KNN", "Random Forest"))

##Get the data
X_train, X_test,y_train, y_test, df, classes = return_data(dataset)
st.dataframe(df.sample(n = 5, random_state = 1))
st.subheader("Classes")

for idx, value in enumerate(classes):
  st.text("{}: {}".format(idx, value))

## 2-PCA 
df = getPCA(df)
fig = plt.figure(figsize = (16, 10))
sns.scatterplot( x = 'pca_1', y = 'pca_2', hue = "Class", palette= sns.color_palette("hls", len(classes)), data = df, legend= "full")
plt.xlabel("PCA One")  
plt.ylabel("PCA Two")
plt.title("2-D PCA Visualization")
st.pyplot(fig)


##Now train the model and get the train, test accuracy scores
model = getClassifier(classifier)
model.fit(X_train, y_train)

train_score = round(model.score(X_train, y_train), 2)
test_score = round(model.score(X_test, y_test),2)

st.subheader("Train Score : {}".format(train_score))
st.subheader("Test Score : {}".format(test_score))