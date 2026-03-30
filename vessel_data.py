import streamlit as st
import pandas as pd

data = pd.read_excel("vessel_input_cleaned.xlsx")
st.title("Data display")
st.dataframe(data)
st.bokeh(data["PortName"])