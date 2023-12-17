from flask import Flask, request, jsonify
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from gensim.utils import simple_preprocess
import gensim.downloader as api
from flask_cors import CORS
import dropbox
from io import BytesIO
import pandas as pd

app = Flask(__name__)

# Enable CORS
CORS(app, resources={r"*": {"origins": "exp://192.168.74.112:8081"}})

# Load or initialize the Word2Vec model
model_word2vec = api.load("word2vec-google-news-300")

user_article_title = None


@app.route("/recommend", methods=["POST", "GET"])
def recommend_articles():
    global user_article_title
    if request.method == "POST":
        user_article_title = request.json.get("article_title")
        if not user_article_title:
            return jsonify({"error": "Article title is required"}), 400
        else:
            return jsonify({"message": "Title received"})

    if request.method == "GET":
        if user_article_title is None:
            return jsonify({"error": "No article title received"}), 400

        # Download data.csv file from Dropbox
        dbx = dropbox.Dropbox(
            "sl.Br6euBNUFQUwCEIh0cLW15-l_bZ9NrZ6AA6-gO0elcyT29PaYKHELQbaiExf4uYsc0QSB5rPiYnY8R0EEqP2BtfQoyyRJFwYswin6qTYV9lskudnh-k0UIpeLKT9MZHIhTQWlJsz2g_I"
        )
        file_path = "/data.csv"
        _, res = dbx.files_download(file_path)
        data = res.content

        # Read data from Dropbox content
        df = pd.read_csv(BytesIO(data))

        # Ensure "title" column contains string values
        df["title"] = df["title"].astype(str)

        # Handle missing or non-string values in the "description" column
        df["description"] = df["description"].fillna("")
        df["description"] = df["description"].astype(str)

        # Tokenize content for Word2Vec using simple_preprocess
        df["tokenized_source"] = df["description"].apply(
            lambda text: simple_preprocess(text)
        )

        # Calculate word vectors for user input title
        tokenized_user_title = simple_preprocess(user_article_title)
        user_title_vector = np.mean(
            [
                model_word2vec[word]
                for word in tokenized_user_title
                if word in model_word2vec.key_to_index
            ],
            axis=0,
        )

        # Calculate similarity scores between user title and articles' titles
        similarity_scores_titles = [
            cosine_similarity(
                user_title_vector.reshape(1, -1),
                np.mean(
                    [
                        model_word2vec[word]
                        for word in tokens
                        if word in model_word2vec.key_to_index
                    ],
                    axis=0,
                ).reshape(1, -1),
            )[0][0]
            if tokens and all(word in model_word2vec.key_to_index for word in tokens)
            else 0
            for tokens in df["tokenized_source"]
        ]

        # Get the indices of top similar articles based on title similarity scores
        similar_articles_indices = sorted(
            range(len(similarity_scores_titles)),
            key=lambda i: similarity_scores_titles[i],
            reverse=True,
        )[:10]

        recommended_articles = [
            {
                "index": i,
                "title": df.loc[i, "title"],
                "source_name": df.loc[i, "source_name"],
                "url": df.loc[i, "url"],
                "published_at": df.loc[i, "published_at"],
                "url_to_image": df.loc[i, "url_to_image"],
            }
            for i in similar_articles_indices
        ]

        return jsonify({"recommended_articles": recommended_articles})


if __name__ == "__main__":
    app.run(debug=True)
