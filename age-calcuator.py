import streamlit as st
from datetime import date as dt

st.title("Age calculator")
today = dt.today()

years = st.date_input("Enter your birth year: ", value=dt(2000,1,1))

st.write("you are below years old: ")
age = today.year - years.year
st.subheader(f"Here's your age: {age}")

