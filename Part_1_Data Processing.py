# !pip install pandas    
# !pip install matplotlib

import pandas as pd
import nltk
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split

# Importér nødvendige biblioteker

# Download nødvendige modeller til tokenisering og stopord
nltk.download('punkt')
nltk.download('stopwords')

# Importér funktioner fra NLTK

# Task 1: Dataindlæsning og -rengøring

def load_data(url):
    """Load data from a CSV file URL."""
    return pd.read_csv(url)

def clean_text(text):
    """Clean text by removing HTML tags, special characters, etc."""
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove digits
    text = re.sub(r'\d+', '', text)
    return text

# Tekstforbehandling

def preprocess_text(text):
    """Preprocess text by tokenizing and converting to lowercase."""
    text = clean_text(text)
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    return tokens

def remove_stopwords(tokens):
    """Remove stopwords from tokens."""
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

def apply_stemming(tokens):
    """Apply stemming to tokens."""
    ps = PorterStemmer()
    return [ps.stem(word) for word in tokens]

# Task 4: Datasætsopdeling

def split_dataset(data):
    """Split dataset into train, validation, and test sets (80/10/10)."""
    # First split: 80% train, 20% temp
    train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
    # Second split: divide temp into val and test (10% each of original)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    return train_data, val_data, test_data

# Task 2: Apply processing pipeline to larger dataset
def process_large_dataset(file_path):
    """Process the larger 995K dataset."""
    print("Loading large dataset...")
    # For large CSV files, consider using chunksize parameter
    large_data = pd.read_csv(file_path)
    
    print(f"Loaded dataset with {len(large_data)} rows")
    
    # Apply same preprocessing as before
    # You may need to optimize for the larger dataset
    return large_data

# Task 3: Data exploration functions
def explore_data(data):
    """Perform exploratory analysis on the dataset."""
    # Basic statistics
    print("\n--- Basic Dataset Statistics ---")
    print(f"Dataset shape: {data.shape}")
    print(f"Columns: {data.columns.tolist()}")
    print(f"Missing values:\n{data.isnull().sum()}")
    
    # Count URLs in content
    def count_urls(text):
        if isinstance(text, str):
            # Simple URL pattern
            url_pattern = r'https?://\S+'
            return len(re.findall(url_pattern, text))
        return 0
    
    # Count dates in content
    def count_dates(text):
        if isinstance(text, str):
            # Simple date patterns (can be expanded)
            date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}'
            return len(re.findall(date_pattern, text))
        return 0
    
    # Count numbers in content
    def count_numbers(text):
        if isinstance(text, str):
            # Find numeric values
            number_pattern = r'\b\d+\b'
            return len(re.findall(number_pattern, text))
        return 0
    
    print("\n--- Content Analysis ---")
    data['url_count'] = data['content'].apply(count_urls)
    data['date_count'] = data['content'].apply(count_dates)
    data['number_count'] = data['content'].apply(count_numbers)
    
    print(f"Average URLs per content: {data['url_count'].mean():.2f}")
    print(f"Average dates per content: {data['date_count'].mean():.2f}")
    print(f"Average numbers per content: {data['number_count'].mean():.2f}")
    
    return data

# Word frequency analysis
def analyze_word_frequency(tokens, tokens_no_stop, tokens_stemmed, top_n=100):
    """Analyze word frequency before and after preprocessing."""
    from collections import Counter
    import matplotlib.pyplot as plt
    
    print("\n--- Word Frequency Analysis ---")
    
    # Original tokens frequency
    freq_original = Counter(tokens)
    most_common_original = freq_original.most_common(top_n)
    print(f"Top {top_n} words before preprocessing:")
    print(most_common_original[:10])  # Show first 10
    
    # After stopword removal
    freq_no_stop = Counter(tokens_no_stop)
    most_common_no_stop = freq_no_stop.most_common(top_n)
    print(f"\nTop {top_n} words after stopword removal:")
    print(most_common_no_stop[:10])  # Show first 10
    
    # After stemming
    freq_stemmed = Counter(tokens_stemmed)
    most_common_stemmed = freq_stemmed.most_common(top_n)
    print(f"\nTop {top_n} words after stemming:")
    print(most_common_stemmed[:10])  # Show first 10
    
    # Plot frequencies
    plt.figure(figsize=(12, 8))
    
    # Get the top 50 words and their frequencies for plotting
    words_original = [word for word, count in most_common_original[:50]]
    counts_original = [count for word, count in most_common_original[:50]]
    
    words_no_stop = [word for word, count in most_common_no_stop[:50]]
    counts_no_stop = [count for word, count in most_common_no_stop[:50]]
    
    words_stemmed = [word for word, count in most_common_stemmed[:50]]
    counts_stemmed = [count for word, count in most_common_stemmed[:50]]
    
    plt.subplot(3, 1, 1)
    plt.bar(range(len(words_original)), counts_original)
    plt.title('Word Frequency (Original)')
    plt.xticks(range(len(words_original)), words_original, rotation=90)
    
    plt.subplot(3, 1, 2)
    plt.bar(range(len(words_no_stop)), counts_no_stop)
    plt.title('Word Frequency (After Stopword Removal)')
    plt.xticks(range(len(words_no_stop)), words_no_stop, rotation=90)
    
    plt.subplot(3, 1, 3)
    plt.bar(range(len(words_stemmed)), counts_stemmed)
    plt.title('Word Frequency (After Stemming)')
    plt.xticks(range(len(words_stemmed)), words_stemmed, rotation=90)
    
    plt.tight_layout()
    plt.savefig('word_frequency_analysis.png')
    plt.close()
    
    print("\nWord frequency plots saved as 'word_frequency_analysis.png'")
    
    return most_common_original, most_common_no_stop, most_common_stemmed

# Hovedfunktion til databehandling
def process_news_data(url):
    """Main function to process news data."""
    # Indlæsning af data
    print("Loading data...")
    data = load_data(url)
    
    # Tekstbehandling og tokenisering
    print("Processing text...")
    all_text = " ".join(data['content'].dropna().tolist())
    
    # Tokenisér og normaliser
    tokens = preprocess_text(all_text)
    
    # Beregn det originale ordforråd
    vocab_original = set(tokens)
    print("Number of unique tokens (before stopword removal):", len(vocab_original))
    
    # Fjernelse af stopord
    tokens_no_stop = remove_stopwords(tokens)
    vocab_no_stop = set(tokens_no_stop)
    print("Number of unique tokens (after stopword removal):", len(vocab_no_stop))
    
    # Beregn reduktionsraten
    reduction_rate_stop = (len(vocab_original) - len(vocab_no_stop)) / len(vocab_original)
    print("Reduction rate after stopword removal:", reduction_rate_stop)
    
    # Stemming
    tokens_stemmed = apply_stemming(tokens_no_stop)
    vocab_stemmed = set(tokens_stemmed)
    print("Number of unique tokens (after stemming):", len(vocab_stemmed))
    
    # Beregn stemming-reduktionsraten
    reduction_rate_stem = (len(vocab_no_stop) - len(vocab_stemmed)) / len(vocab_no_stop)
    print("Reduction rate after stemming:", reduction_rate_stem)
    
    # Udskriv eksempler på tokens
    print("Example tokens (before stemming):", tokens_no_stop[:10])
    print("Example tokens (after stemming):", tokens_stemmed[:10])
    
    # Datasætsopdeling
    print("\nSplitting dataset...")
    train_data, val_data, test_data = split_dataset(data)
    print("Number of rows in training data:", len(train_data))
    print("Number of rows in validation data:", len(val_data))
    print("Number of rows in test data:", len(test_data))
    
    return data, tokens, tokens_no_stop, tokens_stemmed, train_data, val_data, test_data

# Udfør databehandlingen
url = 'https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv'
data, tokens, tokens_no_stop, tokens_stemmed, train_data, val_data, test_data = process_news_data(url)

# Add these calls after your existing code
# For Task 2:
# large_file_path = "path/to/995K_FakeNewsCorpus_subset.csv"  # Update with actual path
# large_data = process_large_dataset(large_file_path)

# For Task 3:
# explored_data = explore_data(data)  # Start with the sample data
# word_freq_results = analyze_word_frequency(tokens, tokens_no_stop, tokens_stemmed)

# When ready to work with the large dataset:
# large_explored_data = explore_data(large_data)
# You'll need to process tokens for the large dataset first