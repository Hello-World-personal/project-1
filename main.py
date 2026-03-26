import streamlit as st

st.title("Hello world")
st.subheader("this is the hope world!")
st.text("Welcome")
st.write("Do you whant to write somthing?")


language = st.selectbox("Pick your favourate programming language: ", ["Python", "Javascript", "Java", "Kotlin"])
st.write(f"{language}, that's an excellent choice")


st.success("Your prefered language is selected successfully")