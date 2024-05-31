import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import os

# Print current working directory for debugging

# Load the dataset with error handling
try:
    bank_data = pd.read_csv("./bank-additional-full.csv", sep=';')
except FileNotFoundError:
    st.error("The dataset file 'bank-additional-full.csv' was not found. Please check the file path.")
    st.stop()

pd.set_option('display.max_columns', None)

org_X = bank_data.drop("y", axis=1)
y = bank_data["y"].map({'no': 0, 'yes': 1})

X = pd.get_dummies(org_X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)

# Implement Logistic Regression
logreg_classifier = LogisticRegression()
logreg_classifier.fit(X_resampled, y_resampled)
logreg_preds = logreg_classifier.predict(X_test_scaled)
indices = [i for i, x in enumerate(logreg_preds) if x == 1][:10]

logreg_accuracy = accuracy_score(y_test, logreg_preds)
logreg_report = classification_report(y_test, logreg_preds)

# Sidebar for theme selection
with st.sidebar:
    theme = option_menu("Choose Theme", ["Light", "Dark"], icons=["sun", "moon"], default_index=0)

if theme == "Dark":
    st.markdown(
        """
        <style>
        body {
            color: #fff;
            background-color: #0e1117;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        body {
            color: #000;
            background-color: #f5deb3; /* wheat color */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

st.title('📃Deposit Prediction Web App')

# Input fields
age = st.number_input('Please enter your age', step=1)
job = st.selectbox('Please select your job', ('admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'))
marital = st.selectbox('Please select your marital status', ('married', 'single', 'divorced', 'unknown'))
education = st.selectbox('Please select your education', ('basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown'))
default = st.selectbox('Please select if you have credit in default or not', ('yes', 'no', 'unknown'))
housing = st.selectbox('Please select if you have a housing loan or not', ('yes', 'no', 'unknown'))
loan = st.selectbox('Please select if you have a personal loan', ('yes', 'no', 'unknown'))
contact = st.selectbox('Please select your contact communication type', ('cellular', 'telephone'))
month = st.selectbox('Please select when was your last contact', ('mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'))
day_of_week = st.selectbox('Please select your last contact day of the week', ('mon', 'tue', 'wed', 'thu', 'fri'))
duration = st.number_input('Please enter your last contact duration in seconds', step=1)
campaign = st.number_input('Please enter the number of contacts performed', step=1)
pdays = st.number_input('Please enter number of days passed after last contacted from a previous campaign', step=1)
if pdays == 0:
    pdays = 999
previous = st.number_input('Please enter the number of contacts performed before this campaign', step=1)
poutcome = st.selectbox('Please select outcome of the previous marketing campaign', ('success', 'failure', 'nonexistent'))
emp_var_rate = st.number_input('Please enter employment variation rate')
cons_price_idx = st.number_input('Please enter consumer price index')
cons_conf_idx = st.number_input('Please enter consumer confidence index')
euribor3m = st.number_input('Please enter euribor 3 month rate')
nr_employed = st.number_input('Please enter number of employees')

if st.button('Press me'):

    with st.spinner('Processing...'):
        input_data = pd.DataFrame({
            'age': [age],
            'job': [job],
            'marital': [marital],
            'education': [education],
            'default': [default],
            'housing': [housing],
            'loan': [loan],
            'contact': [contact],
            'month': [month],
            'day_of_week': [day_of_week],
            'duration': [duration],
            'campaign': [campaign],
            'pdays': [pdays],
            'previous': [previous],
            'poutcome': [poutcome],
            'emp.var.rate': [emp_var_rate],
            'cons.price.idx': [cons_price_idx],
            'cons.conf.idx': [cons_conf_idx],
            'euribor3m': [euribor3m],
            'nr.employed': [nr_employed]
        })

        X = pd.concat([input_data, org_X], ignore_index=True)
        X = pd.get_dummies(X)
        X = X.iloc[[0]]

        X_scaled = scaler.transform(X)
        prediction = logreg_classifier.predict(X_scaled)


    if prediction[0] == 0:
        st.error(f'Client has noot been subscribed to a term deposit.', icon="😢")
    else:
        st.success(f'Client is subscribed to a term deposit.', icon="✅")
