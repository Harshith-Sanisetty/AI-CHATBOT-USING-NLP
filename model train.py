import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from data import data
from preprocessing import preprocess_text

questions = data["questions"]
responses = data["responses"]

preprocessed_questions = [preprocess_text(q) for q in questions]

df = pd.DataFrame({'question': preprocessed_questions, 'response': responses})

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['question'])

y = np.array(df.index)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

def predict_response(user_input):
    user_input_preprocessed = preprocess_text(user_input)
    user_input_vec = vectorizer.transform([user_input_preprocessed])
    response_index = model.predict(user_input_vec)[0]
    return df['response'].iloc[response_index]
