import random

def simulate_events(num_simulations=1000000):

    deck = [f"{rank}{suit}" for rank in "A23456789TJQK" for suit in "CDHS"]
    results = {
        'at_least_one_club_in_first_4': 0,
        'exactly_one_club_in_first_7': 0,
        'first_3_same_suit': 0,
        'first_3_all_sevens': 0,
        'first_5_is_straight': 0
    }

    num_straight = 0
    
    for _ in range(num_simulations):

        random.shuffle(deck)
        first_4 = deck[:4]
        first_7 = deck[:7]
        first_3 = deck[:3]
        first_5 = deck[:5]

        # at least one club in the first four cards
        found_club = False
        for card in first_4:
            if card[-1] == "C":
                found_club = True
                break
        if found_club:
            results['at_least_one_club_in_first_4'] += 1
        
        # exactly one club in the first seven cards
        club_count = 0
        for card in first_7:
            if card[-1] == "C":
                club_count += 1
        if club_count == 1:
            results['exactly_one_club_in_first_7'] += 1
        
        # first three cards are all the same suit
        suit = first_3[0][-1]
        same_suit = True
        for card in first_3:
            if card[-1] != suit:
                same_suit = False
                break
        if same_suit:
            results['first_3_same_suit'] += 1
        
        # first three cards are all sevens
        all_sevens = True
        for card in first_3:
            if card[0] != "7":
                all_sevens = False
                break
        if all_sevens:
            results['first_3_all_sevens'] += 1
        
        # first five cards form a straight, without being a flush
        rank_order = "A23456789TJQK"
        
        rank_values = [rank_order.index(card[0]) for card in first_5]
        rank_values.sort() #comment this line to check the case where also pick order matters
        

        is_straight = True
        for i in range(4):
            if rank_values[i + 1] != rank_values[i] + 1:
                is_straight = False
                break

        # special case where the straight is formed with the ace in last position (10-J-Q-K-A)
        #special_case = False
        if not is_straight:
            if rank_values == [0, 9, 10, 11, 12]: 
                is_straight = True
                #special_case= True

        #if is_straight:
        #    num_straight += 1

        suit_set = set()
        for card in first_5:
            suit_set.add(card[-1])
        
        all_same_suit = len(suit_set) == 1
        if is_straight and not all_same_suit:
            results['first_5_is_straight'] += 1
    
    # probability estimation
    probabilities = {event: count / num_simulations for event, count in results.items()}
    print(f"tot num of straight is {num_straight}/{num_simulations}")
    return probabilities

simulated_probabilities = simulate_events()
print(simulated_probabilities)

