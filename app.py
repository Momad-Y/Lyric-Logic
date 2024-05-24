import streamlit as st
import keras.models as models


st.title("Lyric Logic 🎶")
st.write(
    "This is a simple web app that uses a neural networks to predict the artist based on the lyrics of a song."
)

# Set sidebar size
st.markdown("<style>.css-1l02zno{width: 300px;}</style>", unsafe_allow_html=True)

selected_model = st.sidebar.selectbox(
    "Select the model to use", ["Model 1", "Model 2", "Model 3"]
)
st.sidebar.write("## Artists")
st.sidebar.write("The model can predict the following artists:")
st.sidebar.write(
    "1. Eminem\n2. Taylor Swift\n3. Drake\n4. Ed Sheeran\n5. Beyonce\n6. Rihanna\n7. Lady Gaga\n8. Justin Bieber\n9. Coldplay\n10. Katy Perry\n11. Nicki Minaj\n12. Ariana Grande\n13. Ed Sheeran\n14. Dua Lipa"
)


st.write("## Enter the lyrics")
lyrics = st.text_area("Enter the lyrics of the song you want to predict", height=100)
# Create a button to predict to select the model to use
