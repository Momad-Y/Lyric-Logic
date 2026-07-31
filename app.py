import streamlit as st
import keras.models as models
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from keras.utils import (
    pad_sequences,
)
import pickle
from PIL import Image

nltk.download("stopwords")
nltk.download("wordnet")

artists = [
    "Eminem a.k.a. Slim Shady a.k.a. Marshall Mathers",
    "Taylor Swift a.k.a. Kanye's nemesis",
    "Drake a.k.a. BBL Drizzy",
    "Beyonce a.k.a. Jay-Z's wife",
    "Rihanna a.k.a. Eminem's love interest",
    "Lady Gaga",
    "Justin Bieber a.k.a. the baby",
    "Coldplay",
    "Katy Perry a.k.a. the one with the fireworks",
    "Nicki Minaj",
    "Ariana Grande",
    "Ed Sheeran",
    "Dua Lipa a.k.a. elfnana",
]

artists_images = [
    "https://raw.githubusercontent.com/Momad-Y/Lyric-Logic/refs/heads/main/imgs/em.jpg",
    "https://raw.githubusercontent.com/Momad-Y/Lyric-Logic/refs/heads/main/imgs/swift.jpg",
    "https://raw.githubusercontent.com/Momad-Y/Lyric-Logic/refs/heads/main/imgs/drake.jpg",
    "https://raw.githubusercontent.com/Momad-Y/Lyric-Logic/refs/heads/main/imgs/beyonce.jpg",
    "https://raw.githubusercontent.com/Momad-Y/Lyric-Logic/refs/heads/main/imgs/rihanna.jpg",
    "https://raw.githubusercontent.com/Momad-Y/Lyric-Logic/refs/heads/main/imgs/lg.jpg",
    "https://raw.githubusercontent.com/Momad-Y/Lyric-Logic/refs/heads/main/imgs/jb.jpg",
    "https://raw.githubusercontent.com/Momad-Y/Lyric-Logic/refs/heads/main/imgs/chris.jpg",
    "https://raw.githubusercontent.com/Momad-Y/Lyric-Logic/refs/heads/main/imgs/kp.jpg",
    "https://raw.githubusercontent.com/Momad-Y/Lyric-Logic/refs/heads/main/imgs/nicki.jpg",
    "https://raw.githubusercontent.com/Momad-Y/Lyric-Logic/refs/heads/main/imgs/ariana.jpg",
    "https://raw.githubusercontent.com/Momad-Y/Lyric-Logic/refs/heads/main/imgs/edsheeran.jpg",
    "https://raw.githubusercontent.com/Momad-Y/Lyric-Logic/refs/heads/main/imgs/dualipa.jpg",
]


punctuation = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()
max_sequence_length = 300

# Get the tokenizer
with open("./models/tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)


def preprocess_lyrics(lyrics):
    lyrics = lyrics.lower()
    lyrics = "".join([l for l in lyrics if l not in punctuation])
    lyrics = lyrics.replace(r"[^a-zA-Z0-9 ]", "")
    lyrics = " ".join([word for word in lyrics.split() if word not in stop_words])
    lyrics = " ".join(lemmatizer.lemmatize(word) for word in lyrics.split())
    lyrics = " ".join(stemmer.stem(word) for word in lyrics.split())

    lyrics = [lyrics]
    lyrics = tokenizer.texts_to_sequences(lyrics)
    lyrics = pad_sequences(lyrics, maxlen=max_sequence_length)

    return lyrics


# Load all models at startup
@st.cache_resource
def load_models():
    models_dict = {}
    models_dict["CNN with GloVe"] = models.load_model(
        "./models/Song Lyrics Classification CNN Model with GloVe Embeddings.h5",
        compile=False,
    )
    models_dict["CNN with learned embeddings"] = models.load_model(
        "./models/Song Lyrics Classification CNN Model with Learnable Embeddings.h5",
        compile=False,
    )
    models_dict["LSTM with learned embeddings"] = models.load_model(
        "./models/Song Lyrics Classification LSTM Model with Learnable Embeddings.h5",
        compile=False,
    )
    return models_dict


# Set app title and favicon
st.set_page_config(
    page_title="Lyric Logic",
    page_icon="./imgs/notes.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load all models
all_models = load_models()


# Reduce top margin and customize info block styling
st.markdown(
    """
<style>
    .main .block-container {
        padding-top: 1.rem;
    }
    
</style>
""",
    unsafe_allow_html=True,
)

st.title("Lyric Logic 🎵")
st.write(
    "Lyric Logic is an AI-powered app that identifies music artists based on song lyrics, blending language understanding with a love for music."
)

st.sidebar.image("./imgs/logo.png", caption="Lyric Logic Logo", use_column_width=True)

selected_model = st.sidebar.selectbox(
    "Select the model to use",
    ["CNN with GloVe", "CNN with learned embeddings", "LSTM with learned embeddings"],
)
st.sidebar.write("## Artists")
st.sidebar.write("The model can predict the lyrics of the following artists:")
st.sidebar.write(
    "1. Eminem\n2. Taylor Swift\n3. Drake\n4. Beyonce\n5. Rihanna\n6. Lady Gaga\n7. Justin Bieber\n8. Coldplay\n9. Katy Perry\n10. Nicki Minaj\n11. Ariana Grande\n12. Ed Sheeran\n13. Dua Lipa"
)

st.sidebar.markdown("## Made By:")
st.sidebar.markdown("##### **Mohamed Y Abdelnasser**")
st.sidebar.markdown(
    "##### [Github](https://github.com/Momad-Y) | [LinkedIn](https://www.linkedin.com/in/mohamed-y-abdelnasser/) | [Email](mailto:Mohamed.Y.Abdelnasser@gmail.com)"
)
st.sidebar.markdown("## Disclaimer:")
st.sidebar.markdown(
    "##### This app is for educational purposes only. The model is not perfect and the results are not guaranteed to be accurate."
)
st.sidebar.markdown("## [GitHub Repository](https://github.com/Momad-Y/Lyric-Logic)")

# Create two columns for input and results
input_col, results_col = st.columns([3, 2])

with input_col:
    st.write("## Enter the lyrics")
    lyrics = st.text_area(
        "Enter the lyrics of the song you want to predict, or make up some lyrics to see how the model performs.",
        height=270,
    )

    predict_button = st.button("Predict artist", use_container_width=True)

with results_col:
    st.write("## Prediction Results")

    if predict_button and len(lyrics) > 0:
        if len(lyrics) > max_sequence_length:
            st.warning("The lyrics are too long. Please enter a shorter text.")
        else:
            # Show loading spinner
            with st.spinner("Analyzing lyrics..."):
                processed_lyrics = preprocess_lyrics(lyrics)

                # Get the selected model from pre-loaded models
                model = all_models[selected_model]
                prediction = model.predict(processed_lyrics)
                prediction_index = prediction.argmax(axis=1)

                # Display results
                st.image(
                    artists_images[prediction_index[0]],
                    caption=f"Artist: {artists[prediction_index[0]]}",
                    width=300,
                )

                # Show confidence score
                confidence = prediction.max() * 100
                if confidence < 50:
                    st.write(f"Confidence: {100-confidence:.1f}%")
                else:
                    st.write(f"Confidence: {confidence:.1f}%")

    elif predict_button and len(lyrics) == 0:
        st.warning("Please enter some lyrics to predict the artist.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 20px; margin-top: -40px;'>
        <p style='margin: 0; color: #666; font-size: 14px;'>
            🎵 <strong>Lyric Logic</strong> - AI-Powered Artist Identification 🎵<br>
            Built with ❤️ using Streamlit, TensorFlow, and NLTK<br>
            <em>© 2024 Mohamed Y Abdelnasser. All rights reserved.</em>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
