import streamlit as st
from streamlit_embedcode import github_gist
link='https://gist.github.com/SyedAbdulQadirGilani001/85ea1f8ba3fa774b345f55a23b6119e2'
st.write('Embedding a Gist')
github_gist(link)