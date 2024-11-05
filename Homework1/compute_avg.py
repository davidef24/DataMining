file_path = 'beers.txt'

beers_data = {}

with open(file_path, 'r') as file:
    for line in file:

        # split line into [<beer_name>, <review_score>]
        beer_name, review_score = line.strip().split('\t')
        review_score = float(review_score)  #need to convert review_score from string to float to make calculations

        if beer_name not in beers_data:
            beers_data[beer_name] = {'count': 0, 'total_score': 0.0}
        
        beers_data[beer_name]['count'] += 1
        beers_data[beer_name]['total_score'] += review_score

# include only beers with at least 100 reviews and compute the average
filtered_beers = []

for beer_name, data in beers_data.items():
    if data['count'] >= 100:
        average_score = data['total_score'] / data['count']
        filtered_beers.append((beer_name, average_score))

# sort and get the top 10 beers by average score
top_10_beers = sorted(filtered_beers, key=lambda x: x[1], reverse=True)[:10]

for beer_name, avg_score in top_10_beers:
    print(f'{beer_name}: {avg_score:.2f}')
