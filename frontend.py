import streamlit as st
import  pickle
import numpy as np
import pandas as pd
pipe=pickle.load(open('SalModel.pkl','rb'))
df=pickle.load(open('saldf.pkl','rb'))
st.title("Salary Predictions")
age=st.number_input("Enter the age ",min_value=18 ,max_value=100)
educationlevel=st.selectbox("Enter the Education level ",df['Education Level'].unique())
jobtittle=st.selectbox("Enter the Job Tittle ",df['Job Title'].unique())
expereince=st.number_input("Enter the Year of Experience ")
genders=st.selectbox("Enter the Gender Male/Female",['Male','Female'])
if st.button("Predict salary"):
    if genders=='Male':
        genders=1
    else:
        genders=0
    input_data = pd.DataFrame([{
    "Age": age,
    "Education Level": educationlevel,
    "Job Title":jobtittle,
    "Years of Experience":expereince,
    "Genders": genders
    
}])
    st.markdown(
    f"""
    <div style="
        background-color:#d4edda;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #c3e6cb;
        color: #155724;
        font-size: 25px;
        width: 80%;
        margin: auto;
        text-align: center;
    ">
        ✅ Predicted Salary: ₹{int(pipe.predict(input_data)[0])}
    </div>
    """,
    unsafe_allow_html=True
)
  
    