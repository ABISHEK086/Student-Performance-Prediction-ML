import streamlit as st
import pandas as pd
import seaborn as sns  
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split  
from sklearn.metrics import accuracy_score   
from sklearn.tree import DecisionTreeClassifier     
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression, Perceptron      
from sklearn.neural_network import MLPClassifier  
from sklearn.preprocessing import LabelEncoder    
     
  
st.title("Student Performance Prediction - ML") 
st.write("Upload your dataset or enter details to predict student performance.")
 
@st.cache_data
def load_data(): 
    return pd.read_csv("AI-Data.csv") 

df = load_data()
st.subheader("Dataset Preview")
st.dataframe(df.head())



df_encoded = df.copy()
label_encoders = {}
for col in df_encoded.columns:
    if df_encoded[col].dtype == "object":
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le

target_col = st.selectbox("Select target column (label):", df.columns) 

X = df_encoded.drop(columns=[target_col])
y = df_encoded[target_col]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


models = { 
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Perceptron": Perceptron(),
    "Neural Network (MLP)": MLPClassifier(max_iter=500),
}


st.subheader("Model Accuracy") 
results = {}
for name, model in models.items():
    try:
        model.fit(X_train, y_train) 
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = round(acc * 100, 2)
    except Exception as e:
        results[name] = f"Error: {e}" 

st.write(results)

st.subheader("Visualization") 
col = st.selectbox("Select column to visualize:", df.columns)
fig, ax = plt.subplots()
sns.countplot(data=df, x=col)
st.pyplot(fig)


st.subheader("Predict Student Performance") 


countries = [
    "India", "USA", "UK", "Canada", "Australia", "Germany", "France",
    "China", "Japan", "Brazil", "South Africa", "UAE", "Saudi Arabia",
    "Singapore", "Malaysia", "Italy", "Spain", "Russia", "Mexico", "Other"
]


grades = [f"G-{str(i).zfill(2)}" for i in range(1, 13)] 

subjects = ["Math", "English", "Science", "History", "Geography",
            "Computer", "Biology", "Chemistry", "Physics", "Art", "Other"]

gender = st.selectbox("Gender", ["Male", "Female"])

nationality = st.selectbox("Nationality", countries)
if nationality == "Other":
    nationality = st.text_input("Enter Nationality:")

placeofbirth = st.selectbox("Place of Birth", countries)
if placeofbirth == "Other":
    placeofbirth = st.text_input("Enter Place of Birth:")

stage = st.selectbox("Stage", ["Lower", "Middle", "High"])
grade = st.selectbox("Grade", grades)
section = st.selectbox("Section", ["A", "B", "C"])

topic = st.selectbox("Topic", subjects)
if topic == "Other":
    topic = st.text_input("Enter Subject:") 

relation = st.selectbox("Parent responsible", ["Father", "Mother"])
raisedhands = st.number_input("Raised Hands", min_value=0)
visited = st.number_input("Visited Resources", min_value=0)
discussion = st.number_input("Discussion Participation", min_value=0)
absence = st.number_input("Absence Days", min_value=0)

if st.button("Predict"):
    st.success("Prediction feature will be implemented here with the trained model.")
