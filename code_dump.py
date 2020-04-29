# CS581 Database Management System Ride Sharing Project

import time
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests


class GetCutPoints:
    """ Get the index number of the first people in each time window"""
    
    def __init__(self): 
        self.timewindows, self.cutpoints = self.get_cutdown_points()
        # self.plot_request_desnity()
    
    def get_cutdown_points(self):
        # Time windows: 2min, 5min, 7min, 10min
        # timewindows = [timedelta(minutes=2), timedelta(minutes=5), timedelta(minutes=7), timedelta(minutes=10)]
        timewindows = [timedelta(minutes=5), timedelta(minutes=10)]
        cutpoints = [[] for _ in range(len(timewindows))]
        df = pd.read_csv("df_sorted_jan_pickup.csv")
        # print(df)
        self.df_request_time = df[['tpep_pickup_datetime']]
        # print(self.df_request_time)
        # print(self.df_request_time.loc[0, 'tpep_pickup_datetime'])
        currenttimes = []
        for ind, timewindow in enumerate(timewindows): 
                currenttimes.append(datetime.strptime("2015-01-01 00:00:00", '%Y-%m-%d %H:%M:%S') + timewindow)
                
        for index, row in self.df_request_time.iterrows(): 
            time = datetime.strptime(row['tpep_pickup_datetime'], '%Y-%m-%d %H:%M:%S')
            for ind, timewindow in enumerate(timewindows): 
                
                if time >= currenttimes[ind]:
                    currenttimes[ind] += timewindow
                    if time < currenttimes[ind]:
                        cutpoints[ind].append(index)
                    else: 
                        while time >= currenttimes[ind]: 
                            currenttimes[ind] += timewindow
        cutpoints_temp = [[] for _ in range(4)]
        for i, cutpoint in enumerate(cutpoints):             
            cutpoint.append(index)
            for j, cut in enumerate(cutpoint):
                if cut == cutpoint[-1]:
                    cutpoints_temp[i].append(cut)
                elif cut + 1 != cutpoint[j+1]:
                    cutpoints_temp[i].append(cut)
        cutpoints = cutpoints_temp
                
        return timewindows, cutpoints
    
    def plot_request_desnity(self): 
        df = pd.read_csv("df_sorted_jan_pickup.csv")
        self.df_request_time = df[['tpep_pickup_datetime', "RateCodeID"]]
        self.df_request_time.rename(columns = {"tpep_pickup_datetime":'Pickup_Time'}, inplace = True) 
        self.df_request_time.rename(columns = {"RateCodeID":'Count'}, inplace = True) 
        self.df_request_time['Pickup_Time'] = pd.to_datetime(self.df_request_time['Pickup_Time'])
        self.df_request_time = self.df_request_time.set_index('Pickup_Time')
        # print(self.df_request_time.groupby([self.df_request_time.index.date]))
        
        plot1 = self.df_request_time.groupby([self.df_request_time.index.date]).count().plot(kind='bar', figsize=(10,7))
        fig1 = plot1.get_figure()
        fig1.savefig("4.png", dpi=300)
        plt.show()
        plot2 = self.df_request_time.groupby([self.df_request_time.index.hour]).count().plot(kind='bar', figsize=(10,7))
        fig2 = plot2.get_figure()
        fig2.savefig("5.png", dpi=300)
        plt.show()  
    
    def return_cutpoints(self):
        return self.cutpoints
    
class ReadDataFile:
    def __init__(self):
        df = pd.read_csv("df_sorted_jan_pickup.csv")
        self.df_cor=df[['dropoff_latitude','dropoff_longitude']]
        self.df_cor.rename(columns = {'dropoff_latitude':'latitude'}, inplace = True) 
        self.df_cor.rename(columns = {'dropoff_longitude':'longitude'}, inplace = True) 
        # Drop invalid datapoints
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
    
    def return_df_cor(self):
        return self.df_cor
        
class LoadData: 
    """ Load the coordinates data and the triptime data"""
    
    def __init__(self, df_cor, firstindex, lastindex):
        self.firstindex = firstindex # First datapoint in this timewindow
        self.lastindex = lastindex  # Last datapoint in this timewindow
        
        self.df_cor = df_cor
        self.df_time = None
        self.laguardia = [40.7769, -73.874]
        self.url = "https://graphhopper.com/api/1/matrix"
        
        self.get_cor_data()    # Load coordinates data into dataframe
        self.get_time_data()    # Get the corresponding time matrix from graphhopper
        # self.load_time_data()
        # self.plot_points()
    
    def get_cor_data(self):
        self.df_cor=self.df_cor[self.firstindex:self.lastindex]
        # print(self.df_cor)
        # BBox = ((self.df_cor.longitude.min(), self.df_cor.longitude.max(), self.df_cor.latitude.min(), self.df_cor.latitude.max()))
        # (-79.4538803100586, -72.51884460449219, 39.81238174438477, 41.63334655761719) for all datapoints
        # print(BBox)
        
    def get_time_data(self): 
        lat_long=self.df_cor.values.tolist()
        lat_long=[self.laguardia]+lat_long
        lat_long_list = ["{}, {}".format(i[0], i[1]) for i in lat_long]
        # print(lat_long_list)
        Dict = {"type":"json","vehicle":"car","debug":"true","out_array":["weights","times","distances"],"key":"b43d5095-eb45-421b-965a-b67ead31a572"}
        Dict.update( {"point" : lat_long_list} )
        # print(Dict.get("point"))
        while True: 
            try: 
                response = requests.request("GET", self.url, params=Dict)

                parsed = json.loads(response.text)
                self.df_time = pd.DataFrame(parsed["times"])
                # print(json.dumps(parsed, indent=4, sort_keys=True))
                break
            except KeyError: 
                print("Timeout. Retrying...")
                time.sleep(3)
      
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
    
    def return_df(self):
        return self.df_cor, self.df_time
        
    
class RideSharingAlgorithm:
    """Apply constraints and find the ride sharing plan (rsp)"""
    
    def __init__(self, df_cor, df_time, algorithm=0):
        self.df_cor, self.df_time = df_cor, df_time
        self.num_data = len(self.df_cor)
        self.laguardia = [40.7769, -73.874]
        
        self.delays = 5*60 # In unit of second
        self.possible_shares = {}   # Key is a string. For example, "001002001" means a possible ride sharing between 001 and 002, with going to 001 first. Value is the time saved. 
        
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
        # print("All possible ride sharings: ")
        # print(self.possible_shares)
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
        
        # print("Total time saved: {:.1f}min. ".format(self.total_saved_max/60))
        # print("People participated in the shared rides{}".format(self.shared_people_max))
        # print("Shared rides: {}".format(self.shared_rides_max))
        # print('For example, "001002001" means a possible ride sharing between 001 and 002, with going to 001 first')
        
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
    
    def return_report(self):
        return self.total_saved_max/60, len(self.shared_rides_max)



    
class Batchprocess:
    def __init__(self, df_cor, cutpoints): 
        self.df_cor = df_cor
        self.cutpoints = cutpoints
        self.cut_windows = [5, 10]
        self.saved_time_totals = []
        self.saved_rides_totals = []
        
        for ind_window in range(len(self.cut_windows)):
            saved_time_total, saved_rides_total = 0, 0
            cutpoint = self.cutpoints[ind_window][0:int(60*24/self.cut_windows[ind_window])]    # All cutpoints for one day
            for i, cut in enumerate(cutpoint): 
                if i == 0:
                    self.df_cor_new, self.df_time_new = LoadData(self.df_cor, 0, cut).return_df()
                else: 
                    self.df_cor_new, self.df_time_new = LoadData(self.df_cor, cutpoint[i-1], cut).return_df()
                    
                saved_time, saved_rides = RideSharingAlgorithm(self.df_cor_new, self.df_time_new, algorithm=0).return_report()
                saved_time_total += saved_time
                saved_rides_total += saved_rides
                print("Total time saved: {:.1f}min. Total number of trips saved: {}. {:.1f}% \r".format(saved_time_total, saved_rides, i/len(cutpoint)*100))
            self.saved_time_totals.append(saved_time_total)              
            self.saved_rides_totals.append(saved_rides_total)  
        print(self.saved_time_totals)           
        print(self.saved_rides_totals)            
                                
                
def main():
    start_time = time.time()
    df_cor = ReadDataFile().return_df_cor()
    cutpoints = GetCutPoints().return_cutpoints()
    after_load_time = time.time()
    Batchprocess(df_cor, cutpoints)
    end_time = time.time()
    print("Time Taken to load data: {:.3f}s".format(after_load_time-start_time))
    print("Time Taken to run algorithm: {:.3f}s".format(end_time-after_load_time))


if __name__ == '__main__':
    main()
