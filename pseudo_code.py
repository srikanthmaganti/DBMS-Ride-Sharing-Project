# DFS algorithm

def RideSharingAlgorithm: 
    all_possible_shares = Calculate_shares (timewindow, constraints)
    
    shared_ids = []
    shared_rides = []
    best_distance_saved = 0
    
    DFS (all_possible_shares, shared_ids, shared_rides, best_distance_saved)

def DFS (all_possible_shares, shared_ids, shared_rides, distance_saved): 
    if all_possible_shares is empty: 
        if distance_saved > best_distance_saved: 
            best_distance_saved = distance_saved
    for ride, A, B, distance in all_possible_shares: 
        shared_ids add A and B
        all_possible_shares remove ride
        shared_rides add ride
        distance_saved + distance
        
        DFS (all_possible_shares, shared_ids, shared_rides, best_distance_saved)
        

# Generate centers

def Generate_Centers: 
    Centers = []
    for coordinate in all_ride_requests.coordinate: 
        if coordinate is not close to any center in Centers: 
            Centers.add(coordinate)