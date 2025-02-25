# Importér nødvendige biblioteker:
import pandas as pd                  # Pandas bruges til at læse og håndtere CSV-data.
import nltk                          # NLTK er et bibliotek til naturlig sprogbehandling.

# Downloader nødvendige modeller til tokenisering og stopord:
nltk.download('punkt')               # Punkt-modellen er nødvendig for tokenisering.
nltk.download('stopwords')           # Stopordslister er nødvendige for at fjerne "støj-ord".
nltk.download('punkt_tab')           # Punkt-modellen er nødvendig for tokenisering.

# Importér de funktioner, vi skal bruge fra NLTK:
from nltk.tokenize import word_tokenize  # Funktion til at opdele tekst i ord (tokens).
from nltk.corpus import stopwords        # Til at hente lister af engelske stopord.
from nltk.stem import PorterStemmer      # PorterStemmer bruges til at reducere ord til deres rodform.

# Henter CSV-filen fra URL'en:
url = 'https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv'
data = pd.read_csv(url)              # Læser CSV-filen ind i en Pandas DataFrame.

# Samler al tekst fra kolonnen 'content' til én stor streng.
# Her fjerner vi eventuelle manglende værdier med dropna() for at undgå fejl.
all_text = " ".join(data['content'].dropna().tolist())

# Tokenisering: Opdel den samlede tekst i individuelle ord.
tokens = word_tokenize(all_text)
# Konverterer alle tokens til små bogstaver for at normalisere dataen.
tokens = [token.lower() for token in tokens]

# Beregner det oprindelige ordforråd (unikke tokens) og udskriv antallet:
vocab_original = set(tokens)
print("Antal unikke tokens (før stopordsfjernelse):", len(vocab_original))

# Fjernelser af stopord:
# Henter den engelske stopordsliste fra NLTK.
stop_words = set(stopwords.words('english'))
# Filtrerer tokens, så stopord fjernes:
tokens_no_stop = [word for word in tokens if word not in stop_words]

# Beregner og udskriver ordforrådet efter fjernelse af stopord:
vocab_no_stop = set(tokens_no_stop)
print("Antal unikke tokens (efter stopordsfjernelse):", len(vocab_no_stop))

# Beregner reduktionsraten efter fjernelser af stopord:
reduction_rate_stop = (len(vocab_original) - len(vocab_no_stop)) / len(vocab_original)
print("Reduktionsrate efter stopordsfjernelse:", reduction_rate_stop)

# Anvender stemming for at reducere ord til deres rodformer:
ps = PorterStemmer()                # Initialiser PorterStemmer.
tokens_stemmed = [ps.stem(word) for word in tokens_no_stop]  # Stem hvert token i den filtrerede liste.

# Beregner og udskriver ordforrådet efter stemminger:
vocab_stemmed = set(tokens_stemmed)
print("Antal unikke tokens (efter stemming):", len(vocab_stemmed))

# Beregner reduktionsraten efter stemmingerne:
reduction_rate_stem = (len(vocab_no_stop) - len(vocab_stemmed)) / len(vocab_no_stop)
print("Reduktionsrate efter stemming:", reduction_rate_stem)

# Ekstra: Printer et eksempel på et par tokens for at se effekten af behandlingerne.
print("Eksempel tokens (før):", tokens_no_stop[:10])                # Før stopordsfjernelse.
print("Eksempel tokens (efter stemming):", tokens_stemmed[:10])     # Efter stemming.

