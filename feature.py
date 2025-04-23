import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Sample raw texts
documents = [
    "Cats are beautiful animals.",
    "Dogs are loyal and friendly.",
    "Cats and dogs can live together peacefully.",
    "Some animals are more intelligent than others."
]

# Step 1: Preprocessing - Tokenization, Stopword Removal, and Stemming
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

processed_docs = []

for doc in documents:
    tokens = nltk.word_tokenize(doc.lower())  # lowercase + tokenize
    filtered_tokens = [stemmer.stem(word) for word in tokens if word.isalpha() and word not in stop_words]
    processed_docs.append(" ".join(filtered_tokens))

print("Processed Documents:")
for doc in processed_docs:
    print(doc)

# Step 2: Feature Extraction using Bag of Words
count_vectorizer = CountVectorizer()
bow_matrix = count_vectorizer.fit_transform(processed_docs)

print("\nBag of Words Feature Names:")
print(count_vectorizer.get_feature_names_out())

print("\nBag of Words Matrix:")
print(bow_matrix.toarray())

# Step 3: Feature Extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(processed_docs)

print("\nTF-IDF Feature Names:")
print(tfidf_vectorizer.get_feature_names_out())

print("\nTF-IDF Matrix:")
print(tfidf_matrix.toarray())
