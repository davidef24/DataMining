import csv
from collections import defaultdict

# Define the paths for the input and output files
input_file_path = "amazon_products.tsv"
output_file_path = "amazon_products_no_duplicates.tsv"

# Create a set to track unique descriptions (first field)
unique_descriptions = set()

# Open the input file and write to the output file
with open(input_file_path, mode="r", encoding="utf-8") as infile, \
        open(output_file_path, mode="w", encoding="utf-8", newline="") as outfile:
    
    reader = csv.reader(infile, delimiter="\t")
    writer = csv.writer(outfile, delimiter="\t")
    
    # Copy the header to the output file
    header = next(reader)
    writer.writerow(header)
    
    # Process each row
    for row in reader:
        description = row[0]  # Get the first field
        if description not in unique_descriptions:
            unique_descriptions.add(description)
            writer.writerow(row)
        else:
            print("Found duplicates")

print(f"File without duplicates has been written to '{output_file_path}'.")
