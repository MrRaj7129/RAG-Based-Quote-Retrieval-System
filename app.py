# streamlit_app/app.py
import streamlit as st
from typing import List, Dict
import utils
import time

st.set_page_config(page_title="Quote Search", layout="wide")
st.title("Quote Search — retrieve English quotes")

st.markdown(
    """
    Enter an English query, the app will search the quote corpus (FAISS + SBERT)
    and return the top-k matching quotes with author and score.
    """
)

# Sidebar controls
with st.sidebar:
    st.header("Search settings")
    top_k = st.slider("Top-k results", min_value=1, max_value=20, value=5)
    device = st.selectbox("Model device", options=["cpu"], index=0)
    st.write("Note: embeddings & index should be present in `data/` folder on the server.")

# Initialize resources (cached)
@st.cache_resource
def init():
    # this loads df, embeddings, model, index
    return utils.init_resources(device=device)

with st.spinner("Loading resources (this may take a little while first time)..."):
    try:
        df, embeddings, embed_model, index = init()
    except Exception as e:
        st.error(f"Failed to load resources: {e}")
        st.stop()

# Search box
query = st.text_input("Enter your English query:", value="inspirational quotes about hope")
if st.button("Search") and query.strip():
    start = time.time()
    with st.spinner("Searching..."):
        results = utils.retrieve_top_k(query, df, embed_model, index, embeddings, k=top_k)
    elapsed = time.time() - start
    st.success(f"Found {len(results)} results in {elapsed:.2f}s")

    # Display results
    for i, r in enumerate(results, start=1):
        st.markdown("---")
        header = f"**{i}.** {r['author']}  —  score: {r['score']:.4f}"
        st.write(header)
        st.markdown(f"> {r['quote']}")
        if r.get("tags"):
            st.caption("Tags: " + ", ".join([str(t) for t in r.get("tags", [])]))
else:
    st.info("Type a query and press Search to retrieve quotes.")
