# streamlit_app/app.py
import streamlit as st

st.set_page_config(page_title="Quote Search App", layout="wide")
st.title("Quote Search App — Setup Test")

st.markdown(
    """
    This is a minimal Streamlit app used to verify the deployment pipeline.
    If you see this page, the app file is present and Streamlit can run it.
    """
)

if st.button("Say hello"):
    st.success("Hello — your app is running!")
