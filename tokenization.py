import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK data files (only once)
nltk.download('punkt')
nltk.download('stopwords')

# Sample text
text = "The weather is really beautiful today and I am enjoying the sunshine."

# 1. Tokenization
tokens = word_tokenize(text)

# 2. Remove stopwords
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]

# 3. Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

# Output results
print("Original Text:", text)
print("Tokens:", tokens)
print("Filtered Tokens (no stopwords):", filtered_tokens)
print("Stemmed Tokens:", stemmed_tokens)
