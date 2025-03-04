# Importér nødvendige biblioteker:
import pandas as pd                  # Pandas bruges til at læse og håndtere CSV-data.
import nltk                          # NLTK er et bibliotek til naturlig sprogbehandling.

# Downloader nødvendige modeller til tokenisering og stopord:
nltk.download('punkt')               # Punkt-modellen er nødvendig for tokenisering.
nltk.download('stopwords')           # Stopordslister er nødvendige for at fjerne "støj-ord".



# Importér de funktioner, vi skal bruge fra NLTK:
from nltk.tokenize import word_tokenize  # Funktion til at opdele tekst i ord (tokens).
from nltk.corpus import stopwords        # Til at hente lister af engelske stopord.
from nltk.stem import PorterStemmer      # PorterStemmer bruges til at reducere ord til deres rodform.

# Henter CSV-filen fra URL'en:
data = pd.read_csv('https://raw.githubusercontent.com/several27/FakeNewsCorpus/master/news_sample.csv')         # Læser CSV-filen ind i en Pandas DataFrame.

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


## Part 4
# Importér nødvendige funktioner til datasplitning:
from sklearn.model_selection import train_test_split  # Bruges til at opdele datasættet

# Antag, at 'data' er en Pandas DataFrame, som indeholder hele dit datasæt
# fx datasættet med FakeNewsCorpus, som er blevet forbehandlet tidligere.

# Først splittes dataen i træningsdata (80%) og en midlertidig del (20%)
train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
# Her får train_data 80% af rækkerne, mens temp_data får de resterende 20%.

# Dernæst splittes den midlertidige del i to lige store dele:
# Én del bliver valideringsdata (10% af den oprindelige data) og én del bliver testdata (10% af den oprindelige data).
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Udskriv antallet af rækker i hvert sæt for at bekræfte opdelingen:
print("Antal rækker i træningsdata:", len(train_data))
print("Antal rækker i valideringsdata:", len(val_data))
print("Antal rækker i testdata:", len(test_data))

# Print-udsagn sikrer, at vi kan se resultatet direkte i konsollen.
