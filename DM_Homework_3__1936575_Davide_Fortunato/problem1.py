import hashlib
import time
import random
import math

# Class for generating k-shingles and hashing them
class Shinglings:
    def __init__(self, k):
        self.k = k  # Length of each shingle (subsequence of text)

    def create_shingles(self, document):
        """
        Create k-shingles from a given document.
        A k-shingle is a substring of length k.
        """
        shingles = set()
        for i in range(len(document) - self.k + 1):
            shingle = document[i:i + self.k]
            shingles.add(shingle)
        return shingles

    def hash_shingles(self, shingles):
        """
        Optional, hash each shingle using SHA-1 to generate unique numeric values.
        """
        hashed_shingles = set()
        for shingle in shingles:
            hashed = hashlib.sha1(shingle.encode('utf-8')).hexdigest()  # Hash the shingle
            hashed_shingles.add(int(hashed, 16))  # Convert hash to integer
        return hashed_shingles


# Class for MinHashing
class MinHashing:
    def __init__(self, num_hashes):
        self.num_hashes = num_hashes  # Number of hash functions to use
        self.max_hash = (1 << 32) - 1  # Maximum hash value (2^32 - 1)
        self.hash_functions = self.generate_hash_functions()  # Generate hash functions

    def generate_hash_functions(self):
        """
        Generate a list of hash functions for MinHashing.
        """
        hash_functions = []
        for i in range(self.num_hashes):
            hash_functions.append(self.hash_family(i))  # Generate parametrized hash functions
        return hash_functions

    def hash_family(self, i):
        """
        Create a hash function parameterized by index i.
        """
        result_size = 8  # Number of bytes for the hash value
        max_len = 20
        salt = str(i).zfill(max_len)[-max_len:]  # Generate a unique salt for the hash function

        def hash_member(x):
            return int(hashlib.sha1((x + salt).encode('utf-8')).hexdigest()[-result_size:], 16)
        return hash_member

    def get_signature(self, hashed_shingles):
        """
        Compute the MinHash signature for a set of hashed shingles.
        A signature is the minimum hash value for each hash function.
        """
        signature = []
        for func in self.hash_functions:
            min_hash = min(func(str(shingle)) for shingle in hashed_shingles)
            signature.append(min_hash)
        return signature


# Class for Locality-Sensitive Hashing (LSH)
class LSH:
    def __init__(self, signatures, r, b):
        self.signatures = signatures  # List of signatures from MinHashing
        self.r = r  # Number of rows per band
        self.b = b  # Number of bands
        self.band_buckets = [{} for _ in range(b)]  # Buckets for each band

    def hash_band(self, band):
        """
        Hash a band (subsection of a signature) into a bucket.
        """
        return hashlib.sha1(''.join(map(str, band)).encode('utf-8')).hexdigest()

    def lsh(self):
        """
        Perform LSH to identify candidate pairs of similar documents.
        """
        candidates = set()
        for doc_id, signature in enumerate(self.signatures):
            for band_idx in range(self.b):
                start = band_idx * self.r  # Start of the current band
                end = start + self.r  # End of the current band
                band = signature[start:end]
                band_hash = self.hash_band(band)  # Compute hash for the band
                bucket = self.band_buckets[band_idx].setdefault(band_hash, [])
                for candidate_id in bucket:
                    if candidate_id != doc_id:
                        candidates.add(tuple(sorted((doc_id, candidate_id))))
                bucket.append(doc_id)
        return candidates


# Class for brute-force similarity checking
class BruteForceNN:
    def __init__(self, shingle_sets):
        self.shingle_sets = shingle_sets  # List of shingle sets for all documents

    def jaccard_similarity(self, set1, set2):
        """
        Compute Jaccard similarity between two sets.
        """
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union

    def find_near_duplicates(self, threshold=0.8):
        """
        Find pairs of documents with Jaccard similarity above the given threshold.
        """
        near_duplicates = set()
        num_docs = len(self.shingle_sets)
        for i in range(num_docs):
            for j in range(i + 1, num_docs):
                sim = self.jaccard_similarity(self.shingle_sets[i], self.shingle_sets[j])
                if sim >= threshold:
                    near_duplicates.add((i, j))
        return near_duplicates


def display_duplicates(documents, duplicates):
    """
    Display duplicate document pairs.
    """
    if not duplicates:
        print("No duplicate documents found.")
        return

    for i in range(len(duplicates)):
        doc1_idx, doc2_idx = list(duplicates)[i]
        print(f"Document 1 (ID {doc1_idx}): {documents[doc1_idx]}")
        print(f"Document 2 (ID {doc2_idx}): {documents[doc2_idx]}")


def main():
    # From the book, find out that threshold should be at (1/b)^(1/r)
    # and the highest the number of bands, the more LSH will be precise. 
    # I have choosen 6625 and 25 but other valid options where also b= 10 and r=9
    k = 10  # given in the assignment
    num_hashes = 6625  # number of hash functions
    threshold = 0.8  # given in the assignment
    r = 25  # rows per band
    b = num_hashes // r  # number of bands

    # Read documents from file
    documents = []
    with open('amazon_products.tsv', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Ignore empty lines
                description = line.split('\t')[0]  # Extract first column
                if len(description) >= k:  # Ensure document is long enough for shingling
                    documents.append(description)

    print(f"Total documents: {len(documents)}")

    # Step 1: Shingling
    shingler = Shinglings(k)
    shingle_sets = []
    for doc in documents:
        shingles = shingler.create_shingles(doc)
        hashed_shingles = shingler.hash_shingles(shingles)
        shingle_sets.append(hashed_shingles)

    # Step 2: MinHashing
    minhasher = MinHashing(num_hashes)
    signatures = []
    for hashed_shingles in shingle_sets:
        signature = minhasher.get_signature(hashed_shingles)
        signatures.append(signature)

    # Step 3: Locality-Sensitive Hashing (LSH)
    lsh = LSH(signatures, r, b)
    start_time = time.time()
    candidate_pairs = lsh.lsh()
    lsh_time = time.time() - start_time
    print(f"LSH candidate pairs: {len(candidate_pairs)}")
    print(f"Time taken by LSH: {lsh_time:.2f} seconds")

    # Verify candidate pairs with Jaccard similarity
    lsh_near_duplicates = set()
    for i, j in candidate_pairs:
        sim = BruteForceNN(shingle_sets).jaccard_similarity(shingle_sets[i], shingle_sets[j])
        if sim >= threshold:
            lsh_near_duplicates.add((i, j))
    print(f"LSH near-duplicate pairs: {len(lsh_near_duplicates)}")

    # Step 4: Brute-force comparison
    brute_force = BruteForceNN(shingle_sets)
    start_time = time.time()
    bf_near_duplicates = brute_force.find_near_duplicates(threshold)
    bf_time = time.time() - start_time
    print(f"Brute-force near-duplicate pairs: {len(bf_near_duplicates)}")
    print(f"Time taken by brute-force: {bf_time:.2f} seconds")

    # Intersection of results
    intersection = lsh_near_duplicates.intersection(bf_near_duplicates)
    print(f"Intersection size: {len(intersection)}")

    # Display duplicate examples
    display_duplicates(documents, intersection)


# Execute the main function
main()
