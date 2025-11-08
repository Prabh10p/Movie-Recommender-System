# 🎬 Movie Recommender System

![Banner](assets/Screenshot 2025-11-08 at 2.56.52 PM.png)

---

## 🚀 Project Overview

The **Movie Recommender System** is an interactive web application built with **Streamlit** and **Python**.  
It recommends movies similar to a selected movie using **Natural Language Processing (NLP)** and a **pre-trained similarity model**.  

Users can select a movie and receive **5 recommendations**, each with its **poster image** displayed neatly side by side.

---

## 🔥 Features

- Select a movie from a dropdown list.
- Get **5 recommended movies** with:
  - Movie title
  - Poster image
- **Robust poster fetching** using the **TMDb API**.
- Responsive layout using **Streamlit columns**.
- Fallback placeholder for missing posters.

---

## 🛠 Tech Stack

- **Python 3.11+**
- **Streamlit** – Web interface
- **Pandas** – Dataset handling
- **Pickle** – Load pre-trained models
- **NLTK** – Natural Language Processing
- **TMDb API** – Fetch movie posters

---

## 📂 Project Structure

Movie-Recommender-System/
│
├── app.py # Streamlit application
├── Artifacts/
│ └── Models/
│ ├── data.pkl # Movie dataset
│ └── sim.pkl # Similarity model
├── assets/
│ ├── banner.png # Banner image for README
│ └── screenshot.png # Screenshot of app
├── requirements.txt # Python dependencies
└── README.md


---

## ⚙️ Installation

1. **Clone the repository**


git clone https://github.com/Prabh10p/Movie-Recommender-System.git
cd Movie-Recommender-System


2. **Create and activate a virtual environment** 
python -m venv env
# macOS/Linux
source env/bin/activate
# Windows
env\Scripts\activate

3. **Install dependencies**
pip install -r requirements.txt

4. **Set your TMDb API key**
# macOS/Linux
export MOVIES_API_TOKEN="YOUR_TMDB_API_KEY"
# Windows
set MOVIES_API_TOKEN="YOUR_TMDB_API_KEY"


5. **Run the app with Streamlit**
streamlit run app.py

# 💡 How it Works
- The app uses a pre-trained similarity model (sim.pkl) to find movies similar to the selected movie.
- For each recommended movie, the app fetches the poster from TMDb API.
- Posters are displayed using Streamlit columns for a neat side-by-side layout.
- If a movie has no poster, a placeholder image is displayed.
## NLP & Model Details:
- Tokenization, Stopword removal, Stemming, Lemmatization
- TF-IDF or Count Vectorizer to encode text data
- Cosine Similarity to find similar movies
- Saved model loaded via Pickle