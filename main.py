# CS581 Database Management System Ride Sharing Project

import time
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt


class LoadData: 
    """ Load the coordinates data and the triptime data"""
    
    def __init__(self):
        self.df_cor, self.df_time = None, None
        self.laguardia = [40.7769, -73.874]
        
        self.load_cor_data()
        self.load_time_data()
        # self.plot_points()
    
    def load_cor_data(self):
        df = pd.read_csv("df_sorted_jan_pickup.csv")
        # print(df)
        self.df_cor=df[['dropoff_latitude','dropoff_longitude']][:13]
        # self.df_cor=df[['dropoff_latitude','dropoff_longitude']]
        self.df_cor.rename(columns = {'dropoff_latitude':'latitude'}, inplace = True) 
        self.df_cor.rename(columns = {'dropoff_longitude':'longitude'}, inplace = True) 
        indexNames = self.df_cor[self.df_cor['latitude'] == 0].index
        self.df_cor.drop(indexNames , inplace=True)
        indexNames = self.df_cor[self.df_cor['latitude'] < 38].index
        self.df_cor.drop(indexNames , inplace=True)
        indexNames = self.df_cor[self.df_cor['latitude'] > 42].index
        self.df_cor.drop(indexNames , inplace=True)
        indexNames = self.df_cor[self.df_cor['longitude'] <(-80)].index
        self.df_cor.drop(indexNames , inplace=True)
        indexNames = self.df_cor[self.df_cor['longitude'] >(-70)].index
        self.df_cor.drop(indexNames , inplace=True)
        # print(self.df_cor)
        # BBox = ((self.df_cor.longitude.min(), self.df_cor.longitude.max(), self.df_cor.latitude.min(), self.df_cor.latitude.max()))
        # (-79.4538803100586, -72.51884460449219, 39.81238174438477, 41.63334655761719) for all datapoints
        # print(BBox)
        
    def load_time_data(self):
        
        data = json.load(open('response.json'))
        self.df_time = pd.DataFrame(data["times"])

        # print(self.df_time)
        # print(self.df_time.loc[0, 1])
        
    def plot_points(self):
        BBox = ((-74.2587, -73.6860, 40.4929, 40.9171))
        # BBox = ((-74.2387, -73.6860, 40.4929, 40.9171))
        ruh_m = plt.imread('New_York_City_District_Map_wikimedia.png')
        ratio = (BBox[3]-BBox[2])/(BBox[1]-BBox[0])
        fig, ax = plt.subplots(figsize = (8,8*ratio))
        # Draw lines between all share requests and Laguardia
        for i in range(12): 
            x, y = [self.laguardia[1], self.df_cor.loc[i, "longitude"]], [self.laguardia[0], self.df_cor.loc[i, "latitude"]]
            ax.plot(x, y, marker = 'o', markersize=0, color='b')
        # Plot all share request points
        ax.scatter(self.df_cor.longitude, self.df_cor.latitude, zorder=1, alpha= 1, c='b', s=10)
        # Plot Laguardia airport as single red dot
        ax.scatter(self.laguardia[1], self.laguardia[0], alpha= 1, c='r', s=20)
        ax.set_title('All ride sharing requests in the first time window')
        ax.set_xlim(BBox[0],BBox[1])
        ax.set_ylim(BBox[2],BBox[3])
        ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= ratio*1.8)
        plt.show()
        # fig.savefig('{}.png'.format("1"), dpi=300)
    
    def export_df(self):
        return self.df_cor, self.df_time
        
    
class RideSharing:
    """Apply constraints and find the ride sharing plan (rsp)"""
    
    def __init__(self, df_cor, df_time, algorithm=2):
        self.df_cor, self.df_time = df_cor, df_time
        self.num_data = len(self.df_cor)
        self.laguardia = [40.7769, -73.874]
        
        self.delays = 5*60 # In unit of second
        self.possible_shares = {}   # Key is a string. For example, "001002001" means a possible ride sharing between 001 and 002, with going to 001 first
        
        self.apply_constraints()
        # self.plot_possible_shares()
        if algorithm == 0: 
            self.search()
        elif algorithm == 1: 
            self.stable_matching()
        elif algorithm == 2: 
            self.k_cluster()
            
        # self.plot_sharing_results()
        
    def apply_constraints(self):
        for i in range(self.num_data):
            for j in range(i+1, self.num_data): # "d" in the variables below stand for destination/origin, in this case is the Laguardia airport
                t_da = self.df_time.loc[0, i+1]
                t_db = self.df_time.loc[0, j+1]
                t_ab = self.df_time.loc[i+1, j+1]
                t_ba = self.df_time.loc[j+1, i+1]
                
                if t_da <= t_ba and t_db <= t_ab:  # First constraint
                    pass
                elif (t_da + t_ab) > (t_db + self.delays) and (t_db + t_ba) > (t_da + self.delays):   # Second constraint
                    pass
                else: # When a ride shaing is possible. Notice that if both going to A and B first works, we'll go to the closet one first. 
                    id1 = "{0:0=3d}{1:0=3d}{2:0=3d}".format(i, j, i)
                    id2 = "{0:0=3d}{1:0=3d}{2:0=3d}".format(i, j, j)
                    
                    if (t_da + t_ab) < (t_db + self.delays) and (t_db + t_ba) < (t_da + self.delays):
                        if t_da <= t_db: 
                            self.possible_shares[id1] = t_db - t_ab
                        else: 
                            self.possible_shares[id2] = t_da - t_ba
                    elif (t_da + t_ab) < (t_db + self.delays): # Can only go to A first
                        self.possible_shares[id1] = t_db - t_ab
                    else: # Can only go to B first
                        self.possible_shares[id2] = t_da - t_ba
        print("All possible ride sharings: ")
        print(self.possible_shares)
        # print(len(self.possible_shares))
    
    def plot_possible_shares(self): 
        BBox = ((-74.2587, -73.6860, 40.4929, 40.9171))
        # BBox = ((-74.2387, -73.6860, 40.4929, 40.9171))
        ruh_m = plt.imread('New_York_City_District_Map_wikimedia.png')
        ratio = (BBox[3]-BBox[2])/(BBox[1]-BBox[0])
        fig, ax = plt.subplots(figsize = (8,8*ratio))
        # Draw lines between all share requests and Laguardia
        for i in range(self.num_data):
            for j in range(i+1, self.num_data): 
                id1 = "{0:0=3d}{1:0=3d}{2:0=3d}".format(i, j, i)
                id2 = "{0:0=3d}{1:0=3d}{2:0=3d}".format(i, j, j)
                if id1 in self.possible_shares or id2 in self.possible_shares: 
                    x, y = [self.df_cor.loc[i, "longitude"], self.df_cor.loc[j, "longitude"]], [self.df_cor.loc[i, "latitude"], self.df_cor.loc[j, "latitude"]]
                    ax.plot(x, y, marker = 'o', markersize=0, color='orange')
        # Plot all share request points
        ax.scatter(self.df_cor.longitude, self.df_cor.latitude, zorder=1, alpha= 1, c='b', s=10)
        # Plot Laguardia airport as single red dot
        ax.scatter(self.laguardia[1], self.laguardia[0], alpha= 1, c='r', s=20)
        ax.set_title('All possible ride sharing pairs (With 5min delay tolerance)')
        ax.set_xlim(BBox[0],BBox[1])
        ax.set_ylim(BBox[2],BBox[3])
        ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= ratio*1.8)
        plt.show()
        # fig.savefig('{}.png'.format("2"), dpi=300)
    
    def search(self):
        total_saved = 0 # Total time (s) saved
        shared_rides = []   # Shared rides
        shared_people = []  # People participated in the shared rides
        self.shared_rides_max = []
        self.shared_people_max = []
        self.total_saved_max = 0
        
        self.depth_first_search(total_saved, shared_rides, shared_people)
        
        print("Total time saved: {:.1f}min. ".format(self.total_saved_max/60))
        print("People participated in the shared rides{}".format(self.shared_people_max))
        print("Shared rides: {}".format(self.shared_rides_max))
        print('For example, "001002001" means a possible ride sharing between 001 and 002, with going to 001 first')
        
    def depth_first_search(self, total_saved, shared_rides, shared_people): 
        more_to_share = False
        for share in self.possible_shares: 
            if int(share[0:3]) not in shared_people and int(share[3:6]) not in shared_people: 
                more_to_share = True
                shared_rides.append(share)
                shared_people.append(int(share[0:3]))
                shared_people.append(int(share[3:6]))
                self.depth_first_search(total_saved+self.possible_shares[share], shared_rides, shared_people)
        if more_to_share is False: 
            if total_saved > self.total_saved_max: 
                self.total_saved_max = total_saved
                self.shared_rides_max = shared_rides
                self.shared_people_max = shared_people
                
    def stable_matching(self):
        pass

    def k_cluster(self):
        pass
    
    def plot_sharing_results(self): 
        BBox = ((-74.2587, -73.6860, 40.4929, 40.9171))
        # BBox = ((-74.2387, -73.6860, 40.4929, 40.9171))
        ruh_m = plt.imread('New_York_City_District_Map_wikimedia.png')
        ratio = (BBox[3]-BBox[2])/(BBox[1]-BBox[0])
        fig, ax = plt.subplots(figsize = (8,8*ratio))
        # Draw lines between all share requests and Laguardia
        for ride in self.shared_rides_max:
            x, y = [self.df_cor.loc[int(ride[0:3]), "longitude"], self.df_cor.loc[int(ride[3:6]), "longitude"]], [self.df_cor.loc[int(ride[0:3]), "latitude"], self.df_cor.loc[int(ride[3:6]), "latitude"]]
            ax.plot(x, y, marker = 'o', markersize=0, color='orange')
        # Plot all share request points
        ax.scatter(self.df_cor.longitude, self.df_cor.latitude, zorder=1, alpha= 1, c='b', s=10)
        # Plot Laguardia airport as single red dot
        ax.scatter(self.laguardia[1], self.laguardia[0], alpha= 1, c='r', s=20)
        ax.set_title('Ridesharing Graph (RSG)')
        ax.set_xlim(BBox[0],BBox[1])
        ax.set_ylim(BBox[2],BBox[3])
        ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= ratio*1.8)
        plt.show()
        # fig.savefig('{}.png'.format("3"), dpi=300)                    
                                
                
def main():
    start_time = time.time()
    df_cor, df_time = LoadData().export_df()
    after_load_time = time.time()
    RideSharing(df_cor, df_time, algorithm=0)
    end_time = time.time()
    print("Time Taken to load data: {:.3f}s".format(after_load_time-start_time))
    print("Time Taken to run algorithm: {:.3f}s".format(end_time-start_time))


if __name__ == '__main__':
    main()
