import streamlit as st
import tensorflow.keras.models as models
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from tensorflow.keras.preprocessing.sequence import (
    pad_sequences,
)
import pickle
from PIL import Image

artists = [
    "Eminem a.k.a. Slim Shady a.k.a. Marshall Mathers",
    "Taylor Swift a.k.a. Kanye's nemesis",
    "Drake a.k.a. Kendrick's number one fan",
    "Beyonce a.k.a. Jay-Z's wife",
    "Rihanna a.k.a. Eminem's love interest",
    "Lady Gaga (Who still listens to her?)",
    "Justin Bieber a.k.a. the baby",
    "Coldplay a.k.a. the one with the yellow stars",
    "Katy Perry a.k.a. the one with the fireworks",
    "Nicki Minaj a.k.a. tbh idrc",
    "Ariana Grande a.k.a. the one with the ponytail",
    "Ed Sheeran a.k.a. the ginger",
    "Dua Lipa a.k.a. elfnana",
]

artists_images = [
    Image.open("./imgs/em.jpg"),
    Image.open("./imgs/swift.jpg"),
    Image.open("./imgs/drake.jpg"),
    Image.open("./imgs/beyonce.jpg"),
    Image.open("./imgs/rihanna.jpg"),
    Image.open("./imgs/lg.jpg"),
    Image.open("./imgs/jb.jpg"),
    Image.open("./imgs/chris.jpg"),
    Image.open("./imgs/kp.jpg"),
    Image.open("./imgs/nicki.jpg"),
    Image.open("./imgs/ariana.jpg"),
    Image.open("./imgs/edsheeran.jpg"),
    Image.open("./imgs/dualipa.jpg"),
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


st.title("Lyric Logic 🎶")
st.write(
    "This is a simple web app that uses a neural networks to predict the artist based on the lyrics of a song."
)

selected_model = st.sidebar.selectbox(
    "Select the model to use",
    ["CNN with GloVe", "CNN with learned embeddings", "LSTM with learned embeddings"],
)
st.sidebar.write("## Artists")
st.sidebar.write("The model can predict the following artists:")
st.sidebar.write(
    "1. Eminem\n2. Taylor Swift\n3. Drake\n4. Beyonce\n5. Rihanna\n6. Lady Gaga\n7. Justin Bieber\n8. Coldplay\n9. Katy Perry\n10. Nicki Minaj\n11. Ariana Grande\n12. Ed Sheeran\n13. Dua Lipa"
)

st.write("## Enter the lyrics")
lyrics = st.text_area("Enter the lyrics of the song you want to predict", height=150)

if st.button("Predict artist") and len(lyrics) > 0:
    left_column, right_column = st.columns(2)

    if len(lyrics) > max_sequence_length:
        with left_column:
            st.write("The lyrics are too long. Please enter a shorter text.")

    lyrics = preprocess_lyrics(lyrics)

    if selected_model == "CNN with GloVe":
        model = models.load_model(
            "./models/Song Lyrics Classification CNN Model with GloVe Embeddings.h5"
        )
    elif selected_model == "CNN with learned embeddings":
        model = models.load_model(
            "./models/Song Lyrics Classification CNN Model with Learnable Embeddings.h5"
        )
    elif selected_model == "LSTM with learned embeddings":
        model = models.load_model(
            "./models/Song Lyrics Classification LSTM Model with Learnable Embeddings.h5"
        )

    prediction = model.predict(lyrics)
    prediction = prediction.argmax(axis=1)

    with left_column:
        st.write(f"The predicted artist is: {artists[prediction[0]]}")
    with right_column:
        st.image(
            artists_images[prediction[0]],
            use_column_width=True,
        )
