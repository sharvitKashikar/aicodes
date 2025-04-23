from sklearn.feature_extraction.text import TfidfVectorizer

# Sample documents
documents = [
    "The sky is blue.",
    "The sun is bright.",
    "The sun in the sky is bright.",
    "We can see the shining sun, the bright sun."
]

# Create TF-IDF Vectorizer object
vectorizer = TfidfVectorizer()

# Compute TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(documents)

# Get the words (features)
feature_names = vectorizer.get_feature_names_out()

# Convert matrix to array form for better readability
dense_matrix = tfidf_matrix.toarray()

# Print results
for doc_idx, doc_vector in enumerate(dense_matrix):
    print(f"\nDocument {doc_idx + 1} TF-IDF:")
    for word_idx, tfidf_score in enumerate(doc_vector):
        if tfidf_score > 0:
            print(f"{feature_names[word_idx]}: {tfidf_score:.4f}")
