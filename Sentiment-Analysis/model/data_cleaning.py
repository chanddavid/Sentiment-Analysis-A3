# data_cleaning.py
import os
import re
import random
import pandas as pd
# from data_clean import clean_review
#Data Cleaning
import re
import os
import nltk
import random
import pandas as pd
import contractions                     # to expand "it's" -> "it is", etc.
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Download required NLTK data (run once)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')      # required for newer NLTK
nltk.download('wordnet')               # for lemmatization (optional)


def clean_review(text):

    text = text.lower()
    text = contractions.fix(text)

    # remove only URLs + HTML
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+|www\S+', '', text)

    # keep punctuation for negation learning
    text = re.sub(r'[^a-zA-Z0-9\s\!\?]', ' ', text)

    text = re.sub(r'\s+', ' ', text).strip()

    # 3.7 Tokenize (split into list of words)
    tokens = word_tokenize(text)
    
    # 3.8 Remove stopwords
    stop_words = set(stopwords.words('english'))

    negations = {'not', 'no', 'never', 'nor', 'without', 'neither', 'none', 'nobody', 'nothing', 'cannot', 'couldn', 'wouldn', 'shouldn', 'won', 'don', 'didn', 'isn', 'aren', 'wasn', 'weren'}
    final_stop_words = stop_words - negations  # keep negation words


    # add extra domain‑specific stopwords if needed
    extra_stopwords = {'br', 'u', 'im', 'dont', 'didnt', 'couldnt', 'wouldnt'}
    stop_words.update(extra_stopwords)
    
    # Filter out stopwords and short tokens (length < 2)
    tokens = [word for word in tokens if word not in final_stop_words and len(word) > 1]
    
    # 3.9 (Optional) Lemmatization – reduces words to base form (running → run)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Return cleaned text as a single string (ready for vectorization)
    return ' '.join(tokens)




def load_reviews_from_folder(folder_path):
    print(f"Loading reviews from foldee")
    reviews = []
    for category in ['books', 'dvd', 'electronics','kitchen_&_housewares']:
        pos_file = os.path.join(folder_path, category, 'positive.review')
        neg_file = os.path.join(folder_path, category, 'negative.review')
        for filepath, lbl in [(pos_file, 1), (neg_file, 0)]:
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                matches = re.findall(r'<review_text>(.*?)</review_text>', content, re.DOTALL)
                for text in matches:
                    reviews.append((text.strip(), lbl))
    return reviews


def load_and_clean():
    print("Loading and cleaning data...")
    # Load all data
    data = load_reviews_from_folder('../data') 
    random.shuffle(data)
    texts, labels = zip(*data)

    # Clean
    cleaned_texts = [clean_review(t) for t in texts]

    # Save to CSV
    df = pd.DataFrame({'text': texts, 'cleaned_text': cleaned_texts, 'label': labels})
    df.to_csv('cleaned_reviews.csv', index=False)
    return df
