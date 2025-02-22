from pyspark import SparkContext
from pyspark.sql import SparkSession
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import Counter
import math
import json


# for this execrise, I highly inspired from the laboratory shown in class

# initialize Spark context
sc = SparkContext('local[*]')

# load stop words and stemmer
stop_words = set(stopwords.words('italian'))
ps = PorterStemmer()

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


#let's build the inverted index

# load the data from the file
file_rdd = sc.textFile("amazon_products.tsv")

# skip the header
header = file_rdd.first()
data_rdd = file_rdd.filter(lambda line: line != header)

# now, zip the data with its index
file_rdd_with_index = data_rdd.zipWithIndex()

# split each line by tab (\t) and extract the first column (description) and the index
descriptions_and_indices_rdd = file_rdd_with_index.map(lambda line: (line[0].split("\t")[0], line[1]))

#print(descriptions_and_indices_rdd.take(5))

# create an RDD where each word is paired with the document index
index = descriptions_and_indices_rdd.flatMap(lambda line: [((line[1], word), 1) for word in line[0].lower().split()])

# reduce by key (document index, word) to get the raw count of words in each document
postings = index.reduceByKey(lambda x, y: x + y)

#print(postings.take(10))

# now, we need to calculate the total number of words in each document
word_count_per_doc = descriptions_and_indices_rdd.map(lambda line: (line[1], len(line[0].split())))

#print(word_count_per_doc.take(10))

# join the raw term frequency with the total word count per document
tf_with_word_count = postings.map(lambda post: (post[0][0], (post[0][1], post[1]))) \
                            .join(word_count_per_doc)

#print(tf_with_word_count.take(10)) 

normalized_tf= tf_with_word_count.map(lambda data: (data[1][0][0], (data[0], data[1][0][1] / data[1][1])))  # Calculating TF

#print(normalized_tf.take(10))

#print(normalized_tf.take(50))

postings_2 = postings.map(lambda post:(post[0][1],1))

#postings_2.take(10)

postings_3 = postings_2.reduceByKey(lambda term1,term2 : term1+term2) #key is the first values it sees

#print(postings_3.take(10)) 

num_lines = data_rdd.count()

idf = postings_3.map(lambda term: (term[0],math.log10(num_lines/term[1])))

#print(idf.take(10)) 

#let's compute the tf-idf

rdd=normalized_tf.join(idf)

#print(rdd.take(10))

tf_idf=rdd.map(lambda entry: (entry[1][0][0],(entry[0],entry[1][0][1],entry[1][1],
                                              entry[1][0][1]*entry[1][1]))).sortByKey()

#print(tf_idf.take(10))

tf_idf_2=tf_idf.map(lambda entry: (entry[0],entry[1][0],entry[1][1],entry[1][2],
                                   entry[1][3]))

spark = SparkSession(sc)

df = tf_idf_2.toDF(["DocumentId","Token","TF","IDF","TF-IDF"]).show()


tf_idf_scores = tf_idf_2.map(lambda term: (term[0], term[1], term[3], term[4]))

#print(tf_idf_scores.take(10))

#let's compute norm

only_tf_idf = tf_idf_scores.map(lambda term: (term[0], term[-1]))

#only_tf_idf.take(10)

squared_tf_idf = only_tf_idf.map(lambda term: (term[0], term[1]**2))

#squared_tf_idf.take(10)

norms = squared_tf_idf.reduceByKey(lambda item1, item2: item1 + item2)

#norms.take(10)

norms_sqrt = norms.map(lambda item: (item[0], math.sqrt(item[1])))

#norms_sqrt.take(10)

tf_idf_scores_2 = tf_idf_scores.map(lambda item: (item[0], (item[1], item[2], 
                                                            item[3])))

#tf_idf_scores_2.take(10)

final_index = tf_idf_scores_2.join(norms_sqrt)

#final_index.take(10)

final_index_flatten = final_index.map(lambda item: (item[1][0][0], item[0], 
                                        item[1][0][1], item[1][0][2], item[1][1]))

#print(final_index_flatten.take(10))

#compute cosine similarity between query and documents in final_index_flatten
def compute_cosine_similarity(query, final_index_flatten, descriptions_and_indices_rdd):
    query = query.lower().split(" ")

    # Filter documents that contain the query terms
    filtered_rdd = final_index_flatten.filter(lambda term: term[0] in query)
    
    # Initialize dictionaries for TF, IDF, and norms
    tfs_query = {}
    candidates = {}
    norms = {}
    idf_query = {}
    query_len = len(query)

    # Compute term frequency (TF) for the query
    for word in query:
        tf_word = query.count(word)
        tfs_query[word] = tf_word / query_len
        
        # Get TF-IDF values for the current word in the filtered RDD
        tf_idf_vals = filtered_rdd.filter(lambda term: term[0] == word).map(lambda term: 
                                                (term[1], term[2], term[3], term[4])).collect()
        
        # Compute TF-IDF scores for each document
        for val in tf_idf_vals:
            docId = val[0]
            idf_score = val[1]
            tf_idf_score = val[2]
            norm = val[3]
            norms[docId] = norm
            idf_query[word] = idf_score

            if docId not in candidates:
                candidates[docId] = {word: [idf_score, tf_idf_score]}
            else:
                candidates[docId][word] = [idf_score, tf_idf_score]

    # Update candidates to include missing words with zero TF-IDF
    for doc in candidates:
        for word in query:
            if word not in candidates[doc]:
                candidates[doc][word] = [idf_query[word], 0]

    # Compute cosine similarity scores
    top_scores = {}

    for doc in candidates:
        query_tf_idf_scores = []
        cumulative_tf_idf = 0

        for word in candidates[doc]:
            query_tf_idf = tfs_query[word] * candidates[doc][word][0]
            query_tf_idf_scores.append(query_tf_idf)
            doc_tf_idf = candidates[doc][word][1] 
            cumulative_tf_idf += query_tf_idf * doc_tf_idf

        query_norm = 0
        for elem in query_tf_idf_scores:
            query_norm += elem**2

        query_norm = math.sqrt(query_norm)

        top_scores[doc] = cumulative_tf_idf / (query_norm * norms[doc])

    #same procedure of the lab
    # sort the documents based on cosine similarity scores and get the top 5
    top5 = list(sorted(top_scores.items(), key=lambda item: item[1], reverse=True))[:5]

    # reverse the document descriptions and indices for final output
    txt_reverse = descriptions_and_indices_rdd.map(lambda line: (line[1], line[0]))

    # combine the top 5 documents with their descriptions and sort by similarity
    results = sc.parallelize(top5, 2).join(txt_reverse).sortBy(lambda item: item[1][0], False).map(lambda el: (el[1][1], el[1][0]))
    
    return results

# Example usage:
query1 = "Custodia per Macbook Air 13 Pollici"
query2 = "AIPIE Custodia per Macbook Air 13 Pollici Proteggi Custodia per Macbook Pro 13 Pollici, Borsa Porta pc Computer Custodia Cover Notebook Laptop 13,3 Pollici Copri 32 x 23 x 2,5 cm"

# Assuming final_index_flatten and descriptions_and_indices_rdd are already defined
results_query1 = compute_cosine_similarity(query1, final_index_flatten, descriptions_and_indices_rdd)
results_query2 = compute_cosine_similarity(query2, final_index_flatten, descriptions_and_indices_rdd)

print(results_query1.collect())
print(results_query2.collect())