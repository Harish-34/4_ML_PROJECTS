import streamlit as st
import pickle
import numpy as np

#Load the pickle model
with open(r'salary_rediction_slm_streamlit/linera_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)
# model = pickle.load(open(r'C:\Users\91837\DSPrakashSenapathi\4_ml\4_ml_projects\salary_rediction_slm_streamlit\linera_regression_model.pkl'))

#set the title of streamlit app
st.title("Salary Prediction App")

#Add a brief description
st.write("This app predicts the salary based on years of experience.")

# Add input wedget for user to enter years of experience
years_experience = st.number_input("Enter years of experience: ",min_value=0.0,max_value=50.0,value=1.0,step=0.5)

# when the button is clicked make predictions
if st.button("Predict Salary"):
    # Make prediction using the trained model
    experience_input = np.array([[years_experience]]) #convert the input number to a 2D array for prediction
    prediction =model.predict(experience_input)

    #Display the result
    st.success(f"The predicted salary for {years_experience} years of experience is: ${prediction[0]:,.2f}")

#display the info about the model
st.write("This model was trained using a dataset of salary and years of experience by Harish34")
