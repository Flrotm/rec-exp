import streamlit as st
from caminob import show_camino_b
from caminoa import show_camino_a


page = st.sidebar.selectbox("Experimento", ("Experimento A", "Experimento B"))

if page == "Experimento A":
    show_camino_a()
else:
    show_camino_b()