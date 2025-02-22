import requests
import time
from bs4 import BeautifulSoup
import csv
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict
import math
import json
import re
import unicodedata


nltk.download('all')


def calculate_tf(doc):
    words = preprocess_text(doc).split()
    word_count = len(words)
    tf = Counter(words)
    for word in tf:
        tf[word] /= word_count  # Normalize by total number of words
    return tf

def calculate_idf(docs):
    N = len(docs)
    idf = defaultdict(lambda: 0)
    
    # Count documents containing each word
    for doc in docs:
        words = set(preprocess_text(doc).split())
        for word in words:
            idf[word] += 1

    # Calculate IDF
    for word in idf:
        idf[word] = math.log(N / (idf[word]))  # Added 1 to avoid division by zero
    return idf

def calculate_tf_idf(docs):
    idf = calculate_idf(docs)
    tf_idf = []
    
    for doc in docs:
        tf = calculate_tf(doc)
        doc_tf_idf = {word: tf[word] * idf[word] for word in tf}
        tf_idf.append(doc_tf_idf)
        
    return tf_idf

def cosine_similarity(vec1, vec2):
    #print(f"{vec1}, {vec2}")
    # Compute dot product
    dot_product = sum(vec1[word] * vec2.get(word, 0) for word in vec1)
    
    # Compute magnitudes
    magnitude1 = math.sqrt(sum(val ** 2 for val in vec1.values()))
    magnitude2 = math.sqrt(sum(val ** 2 for val in vec2.values()))
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)


def fetch_amazon_page(keyword, page):
    url = f"https://www.amazon.it/s?k={keyword}&page={page}"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'}
    response = requests.get(url, headers=headers)
    time.sleep(2)
    return response.text if response.status_code == 200 else None

def parse_products(page_content):
    soup = BeautifulSoup(page_content, 'html.parser')
    products = []
    seen_descriptions = set()  # To track unique descriptions

    for item in soup.find_all('div', {'data-component-type': 's-search-result'}):
        # Extract product details
        description_1 = item.h2.text.strip()
        description = unicodedata.normalize('NFKC', description_1)

        normalized_description = tuple(re.sub(r'\W+', ' ', description.lower()).strip().split())
        #print(normalized_description)
        # Check if the normalized description is a duplicate
        if normalized_description in seen_descriptions:
            #print(f"Duplicate found: {normalized_description}")
            continue
        seen_descriptions.add(normalized_description)

        price = item.find('span', {'class': 'a-price-whole'}).text.strip() if item.find('span', {'class': 'a-price-whole'}) else 'N/A'
        prime = bool(item.find('i', {'aria-label': 'Amazon Prime'}))
        stars = item.find('span', {'class': 'a-icon-alt'}).text.strip() if item.find('span', {'class': 'a-icon-alt'}) else 'N/A'
        url = "https://www.amazon.it" + item.h2.a['href']
        
        products.append([description, price, prime, url, stars])

    return products



def save_to_tsv(products, filename):
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['Description', 'Price', 'Prime', 'URL', 'Stars'])
        for product in products:
            writer.writerow(product)


def preprocess_text(text):
    # load italian stop words
    stop_words = set(stopwords.words('italian'))
    
    # Iinitialize the Porter Stemmer for stemming words
    ps = PorterStemmer()
    
    # tokenize the text into words and convert to lowercase for uniformity
    words = word_tokenize(text.lower())
    
    # remove stop words and non-alphanumeric words, then stem each word
    words = [ps.stem(w) for w in words if w.isalnum() and w not in stop_words]
    
    # join the processed words back into a single string and return
    return ' '.join(words)


from collections import defaultdict

def build_inverted_index(lines):
    # init empty inverted index (structure is word: [<list of line indices where the word is contained>])
    inverted_index = defaultdict(list)
    
    # loop for every line
    for i, (description, *_) in enumerate(lines):
        # preprocess description
        processed_text = preprocess_text(description)
        # split the processed description into individual words and add each to the inverted index
        for word in processed_text.split():
            # append the line index to the list of indices for this word
            inverted_index[word].append(i)

    return inverted_index

def save_inverted_index(index, filename):
    with open(filename, 'w') as file:
        json.dump(index, file)


def search_products(query, products):
    # extract descriptions and calculate TF-IDF for each product
    descriptions = [p[0] for p in products]
    tf_idf_docs = calculate_tf_idf(descriptions)
    
    # compute TF-IDF vector for the query
    idf = calculate_idf(descriptions)
    query_tf = calculate_tf(query)
    query_tf_idf = {word: query_tf[word] * idf.get(word, 0) for word in query_tf}
    
    # compute cosine similarity for each document
    similarities = [(i, cosine_similarity(query_tf_idf, doc_tf_idf)) for i, doc_tf_idf in enumerate(tf_idf_docs)]
    
    # sort by similarity score in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # return the top 5 results
    return [(products[i], score) for i, score in similarities[:5]]

pages = 30  #amazon pages to scrape
keyword = "computer"
all_products = []

for page in range(1, pages + 1):
    #download pages
    content = fetch_amazon_page(keyword, page)
    if content:
        # parse products
        products = parse_products(content)
        all_products.extend(products)

save_to_tsv(all_products, 'amazon_products.tsv')

# do preprocessing and build inverted index
index = build_inverted_index(all_products)
save_inverted_index(index, 'inverted_index.json')


print("############################## QUERY 1 ################################################")

# perform queries
query = "macbook pro m2/m1"
results = search_products(query, all_products)
for product, score in results:
    print(f"{product[0]} - Score: {score}")

print("##############################################################################")
print("############################## QUERY 2 ################################################")

query2 = "zaino impermeabile pc 15 pollici"
results2 = search_products(query2, all_products)
for product, score in results2:
    print(f"{product[0]} - Score: {score}")

print("##############################################################################")
print("############################## QUERY 3 ################################################")

query3 = "AIPIE Custodia per Macbook Air 13 Pollici Proteggi Custodia per Macbook Pro 13 Pollici, Borsa Porta pc Computer Custodia Cover Notebook Laptop 13,3 Pollici Copri 32 x 23 x 2,5 cm"
results3 = search_products(query3, all_products)
for product, score in results3:
    print(f"{product[0]} - Score: {score}")

print("##############################################################################")
