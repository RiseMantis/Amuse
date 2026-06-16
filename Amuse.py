import streamlit as st
import pandas as pd
import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL", 'http://localhost:8000')

mdText = """
# Amuse
### Your Walmart Music Recommender
*"Hey I like that record"*
"""

st.write(mdText)

song = st.text_input("Enter Song Name")

res = {}

if st.button("Submit"):
  payload = {'song': song}
  try:
    res = requests.post(f"{BACKEND_URL}/backend", json=payload)
    res.raise_for_status()
    
  except:
    st.error("Request Failed")

if res:
  df = pd.DataFrame(res.json())
  st.dataframe(df)
  