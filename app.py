import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
import joblib

# Set page title
st.title("Credit Card Fraud Detection")

# Upload dataset
uploaded_file = st.file_uploader("Upload your credit card dataset", type=["csv"])

if uploaded_file is not None:
    # Load data
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(data.head())

    # Preprocess data
    st.write("Preprocessing the data...")
    sc = StandardScaler()
    data['Amount'] = sc.fit_transform(pd.DataFrame(data['Amount']))
    data = data.drop(['Time'], axis=1)
    data = data.drop_duplicates()

    X = data.drop('Class', axis=1)
    y = data['Class']

    # Handle imbalanced data with SMOTE
    X_res, y_res = SMOTE().fit_resample(X, y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

    # Choose classifier
    classifier = st.selectbox("Choose classifier", ["Logistic Regression", "Decision Tree Classifier"])
    
    # Initialize model
    if classifier == "Logistic Regression":
        clf = LogisticRegression()
    elif classifier == "Decision Tree Classifier":
        clf = DecisionTreeClassifier()

    # Train the model
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Display evaluation metrics
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    st.write(f"F1 Score: {f1_score(y_test, y_pred)}")
    st.write(f"Precision Score: {precision_score(y_test, y_pred)}")
    st.write(f"Recall Score: {recall_score(y_test, y_pred)}")

    # Save the model
    joblib.dump(clf, "credit_card_model.pkl")
    st.write("Model saved as 'credit_card_model.pkl'")

    # Predict a sample transaction
    st.write("Test with a sample transaction:")
    sample_transaction = st.text_input("Enter transaction features (comma-separated):", 
                                       "-1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,...")

    if st.button("Predict Transaction"):
        if sample_transaction:
            model = joblib.load("credit_card_model.pkl")
            transaction = np.array(sample_transaction.split(",")).reshape(1, -1).astype(float)
            pred = model.predict(transaction)

            if pred[0] == 0:
                st.success("Normal Transaction")
            else:
                st.error("Fraud Transaction")
        else:
            st.error("Please enter transaction features.")
