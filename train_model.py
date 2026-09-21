import sqlite3 
import pandas as pd 
import joblib 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.pipeline import Pipeline 

DB_PATH = "jobs.db"
MODEL_PATH = "category_model.joblib"

def load_data():
    """Load job title + snippet text and category labels from the database."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT title, snippet, category FROM jobs", conn)
    conn.close()
    return df 

def prepare_text(df):
    """Combine title and snippet into one text field per job - gives the model more words to learn from than title alone."""
    df["text"] = df["title"].fillna("") + " " + df["snippet"].fillna("")
    return df 

def train_model(df):
    """Train a TF-IDF + Logistic Regression pipeline to predict category from job text.
    
    To Note: with only ~27 rows, there isn't enough data for a meaningful train/test split - this model is trained on tehe full dataset and should be understood as a proof of concept, not a full classifier. It is documented as a limitation in the README.
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000)),
    ])

    pipeline.fit(df["text"], df["category"])
    return pipeline 

def main():
    df = load_data()
    df = prepare_text(df)

    print(f"Training on {len(df)} listings")
    print("Category destribution:")
    print(df["category"].value_counts())

    model = train_model(df)

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    print("\nSample predictions (on training data, not held-out test data):")
    sample = df.sample(min(5, len(df)), random_state=42)
    predictions = model.predict(sample["text"])
    for title, actual, predicted in zip(sample["title"], sample["category"], predictions):
        print(f"  '{title}' -> actual: {actual}, predicted: {predicted}")

if __name__ == "__main__":
    main()


