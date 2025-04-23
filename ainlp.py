# Step 1: Import necessary libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Step 2: Prepare sample training data
texts = [
    "I loved the movie, it was amazing!",
    "Fantastic film with a great story.",
    "Terrible movie, I hated it.",
    "The plot was boring and predictable.",
    "What a wonderful experience!",
    "Worst movie I've seen in years.",
    "That was a very enjoyable movie!",
    "The suspense in the film was thrilling.",
    "Not a single exciting moment in it.",
    "Enjoyable and full of surprises."
]
labels = ['pos', 'pos', 'neg', 'neg', 'pos', 'neg', 'pos', 'pos', 'neg', 'pos']  # Labels for sentiment (positive/negative)

# Step 3: Create a model pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Step 4: Train the model
model.fit(texts, labels)

# Step 5: Test with new data
test_text = "The movie was full of suspense and very enjoyable"
prediction = model.predict([test_text])

# Step 6: Show the result
print("Predicted sentiment:", prediction[0])
