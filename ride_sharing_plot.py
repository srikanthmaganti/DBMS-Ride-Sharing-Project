# CS581 Database Management System Ride Sharing Project

import os
import time
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
from ride_sharing import RideSharingAlgorithm
from pathlib import Path


class PreloadData:
    def __init__(self): 
        self.df_center_cor = pd.read_csv("center_points.csv")
        self.full_time_matrix = np.genfromtxt('time_matrix.csv', delimiter=',')
    
    def return_data(self): 
        return self.df_center_cor, self.full_time_matrix
        
class PlotAWindow:
    """Pick a pool window, plot all ride sharing result. """
    
    def __init__(self, year, month, day, hour_start, min_start, pool_size, df_center_cor = "", full_time_matrix = "", savefig = False, batch=False):
        self.total_start_time = time.time()
        self.year = year
        self.month = month
        self.day = day
        self.hour_start = hour_start
        self.min_start = min_start
        self.pool_size = pool_size 
        self.string = ""
        self.batch = batch
        
        self.starttime = datetime.strptime('{:0=2d}:{:0=2d}:00'.format(self.hour_start, self.min_start), '%H:%M:%S')
        self.endtime = self.starttime + timedelta(minutes=self.pool_size)
        
        file1 = "Data/{}_pickup_{}.csv".format(month, year)
        file2 = "Data/{}_dropoff_{}.csv".format(month, year)
        self.df_cor1 = self.readdatafile(file1, "dropoff")
        self.df_cor2 = self.readdatafile(file2, "pickup")
        self.rides_total = [len(self.df_cor1), len(self.df_cor2)]
        self.df_titles = ["pickup", "dropoff"]
        self.dfs = [self.df_cor1, self.df_cor2]
        if type(df_center_cor) == pd.DataFrame:
            self.df_center_cor = pd.read_csv("center_points.csv")
        else: 
            self.df_center_cor = df_center_cor
        self.list_of_centers = []
        self.algorithms = []
        for i in range(len(self.df_center_cor)): 
            row = self.df_center_cor.iloc[i]
            cor = [row["longitude"], row["latitude"]]
            self.list_of_centers.append(cor)
        # These are the limit ranges to find the closest center to each ride share request: 
        self.longlimit = 0.005
        self.latlimit = 0.005
        # Load the time matrix with transportation time between every center pairs: 
        if type(full_time_matrix) == pd.DataFrame:
            self.full_time_matrix = np.genfromtxt('time_matrix.csv', delimiter=',')
        else:
            self.full_time_matrix = full_time_matrix
        
        self.saved_totals = {"saved_time_pickup":0, "saved_rides_pickup":0, "saved_time_dropoff":0, "saved_rides_dropoff":0, 
                        "av_per_saved_time_pickup":0, "av_per_saved_rides_pickup":0, "av_per_saved_time_dropoff":0, "av_per_saved_rides_dropoff":0,
                        "av_computation_time_pickup":0, "av_computation_time_dropoff":0}
        
        self.run()
        self.final_report()
        self.plotall(show = False, save=True)
        
    def run(self):    
        # Wihtout ride sharing, what the time and rides would be: 
        time_total, rides_total = [0, 0], [0, 0]
        
        for df_index, df in enumerate(self.dfs): # One for pickup one for dropoff
            start_time = time.time()
            
            # First, get the coordinate matrix and time matrix for all requests in this time window: 
            df_cor = df[['latitude','longitude']]   # Only keep the coordinates of all requests in a time window. 
            df_time = self.cal_time_matrix(df_cor)
            # Run algorithm
            algorithm = RideSharingAlgorithm(df_cor, df_time, algorithm=0)
            saved_time, saved_rides = algorithm.return_report()
            self.algorithms.append(algorithm)
            # Calculate the total time and total rides without save: 
            for t in df_time[0]: 
                time_total[df_index] += t
            rides_total[df_index] += len(df_cor) 
            # Calculate maga data
            self.saved_totals["saved_time_{}".format(self.df_titles[df_index])] += saved_time
            self.saved_totals["saved_rides_{}".format(self.df_titles[df_index])] += saved_rides
            end_time = time.time()
            average_time = end_time-start_time
            self.saved_totals["av_computation_time_{}".format(self.df_titles[df_index])] = average_time
            self.saved_totals["av_per_saved_time_{}".format(self.df_titles[df_index])] = \
                self.saved_totals["saved_time_{}".format(self.df_titles[df_index])] / time_total[df_index]
            self.saved_totals["av_per_saved_rides_{}".format(self.df_titles[df_index])] = \
                self.saved_totals["saved_rides_{}".format(self.df_titles[df_index])] / rides_total[df_index]

    def final_report(self):
        print("Final report: ")
        print("Test Time: {} {}-{}. ".format(self.string, self.starttime.strftime('%H:%M:%S'), self.endtime.strftime('%H:%M:%S')))
        print("Total trips (from LGA): {}; Total trips (to LGA): {}. ".format(self.rides_total[0], self.rides_total[1]))
        print("For time window (pool size): {}mins: ".format(self.pool_size))
        
        print("For pickup at the airport: Total time saved: {:.1f}min. Total number of trips saved: {}. "\
            .format(self.saved_totals["saved_time_pickup"], self.saved_totals["saved_rides_pickup"]))
        print("For dropoff at the airport: Total time saved: {:.1f}min. Total number of trips saved: {}. "\
            .format(self.saved_totals["saved_time_dropoff"], self.saved_totals["saved_rides_dropoff"]))
        print("For pickup at the airport: Average time saved per pool (%): {:.1f}. Average number of trips saved per pool (%): {:.1f}. "\
            .format(self.saved_totals["av_per_saved_time_pickup"]*100, self.saved_totals["av_per_saved_rides_pickup"]*100))
        print("For dropoff at the airport: Average time saved per pool (%): {:.1f}. Average number of trips saved per pool (%): {:.1f}. "\
            .format(self.saved_totals["av_per_saved_time_dropoff"]*100, self.saved_totals["av_per_saved_rides_dropoff"]*100))
        print("Algorithm Running time: For pickup: {:.2f}s. For dropoff: {:.2f}s. "\
            .format(self.saved_totals["av_computation_time_pickup"], self.saved_totals["av_computation_time_dropoff"]))
        print("*"*60)
        self.total_end_time = time.time()
        print("Time Taken to run everything: {:.2f}s. \n".format(self.total_end_time-self.total_start_time))
    
    def plotall(self, show = True, save=False):
        for df_index, df in enumerate(self.dfs): 
            df_cor = df[['latitude','longitude']]
            title = self.df_titles[df_index]
            
            folder = "{}_{}_to_{}".format(self.string, self.starttime.strftime('%H-%M-%S'), self.endtime.strftime('%H-%M-%S'))
            if save: 
                Path(folder).mkdir(parents=True, exist_ok=True)
            
            self.plot_points(df_cor, title, folder, show, save)
            self.algorithms[df_index].plot_possible_shares(title, folder, show, save)
            self.algorithms[df_index].plot_sharing_results(title, folder, show, save)
    
    def plot_points(self, df_cor, title="pickup", folder="", show = True, save=False): 
        laguardia = [40.7769, -73.874]
        BBox = ((-74.2587, -73.6860, 40.4929, 40.9171))
        # BBox = ((-74.2387, -73.6860, 40.4929, 40.9171))
        ruh_m = plt.imread('New_York_City_District_Map_wikimedia.png')
        ratio = (BBox[3]-BBox[2])/(BBox[1]-BBox[0])
        fig, ax = plt.subplots(figsize = (8,8*ratio))
        # Draw lines between all share requests and Laguardia
        for i in range(len(df_cor)): 
            x, y = [laguardia[1], df_cor.loc[i, "longitude"]], [laguardia[0], df_cor.loc[i, "latitude"]]
            ax.plot(x, y, marker = 'o', markersize=0, linewidth=0.3, color='b')
        # Plot all share request points
        ax.scatter(df_cor.longitude, df_cor.latitude, zorder=1, alpha= 1, c='b', s=10)
        # Plot Laguardia airport as single red dot
        ax.scatter(laguardia[1], laguardia[0], alpha= 1, c='r', marker = 'x', s=80)
        if title == "pickup": 
            title2 = "from"
        else: 
            title2 = "to"
        ax.set_title('All ride sharing requests ({} LGA)'.format(title2))
        ax.set_xlim(BBox[0],BBox[1])
        ax.set_ylim(BBox[2],BBox[3])
        ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= ratio*1.8)
        if show: 
            plt.show()
        if save: 
            fig.savefig(folder + os.sep + 'all_ride_requests_{}.png'.format(title), dpi=300)

    def cal_time_matrix(self, df_cor): 
        """Calcualte the travel time between each pair of requests"""
        
        # Initial the matrix with zeros. Notice index = 0 is also the airport.
        time_matrix = np.zeros((len(df_cor)+1, len(df_cor)+1))
        # First, find the center each request belongs to
        list_of_center_indices = [0]    # 0 is initialized for the airport
        for i in range(len(df_cor)): 
            data = df_cor.iloc[i]
            cor = [data["longitude"], data["latitude"]]
            index = self.return_belonged_center(cor)
            list_of_center_indices.append(index)   
        # Generate the time matrix for all requests in this time window based on the precomputed full time matrix. 
        for i in range(len(df_cor)+1): 
            cor1 = list_of_center_indices[i]
            for j in range(len(df_cor)+1): 
                if i == j: 
                    continue
                cor2 = list_of_center_indices[j]
                
                if cor1 == -1 or cor2 == -1: 
                    time_matrix[i, j] = 200
                else: 
                    time_matrix[i, j] = self.full_time_matrix[cor1, cor2]
        return time_matrix
        
    def return_belonged_center(self, cor): 
        index = -1
        min_distance2 = 10000
        for distance_tolarance in range(10):
            for i, center in enumerate(self.list_of_centers): 
                if abs(center[0]-cor[0]) <= (distance_tolarance + 1) * self.longlimit and abs(center[1]-cor[1]) <= (distance_tolarance + 1) * self.latlimit: 
                    # Find the best center. Cooment out if just want to find a center
                    # distance2 = (center[0]-cor[0]) * (center[0]-cor[0]) + (center[1]-cor[1])*(center[1]-cor[1])
                    # if distance2 < min_distance2: 
                    #     index = i
                    #     min_distance2 = distance2
                    return i 
        # No first option center and no further center can be found
        return -1
    
    def readdatafile(self, file, title): 
        """ Read csv file into dataframe, and filter out invalid data"""
        # pd.reset_option('mode.chained_assignment')
        # with pd.option_context('mode.chained_assignment', None):
        
        BBox = [-74.2587, -73.6860, 40.4929, 40.9171]
          
        df = pd.read_csv(file)
        df.rename(columns = {'{}_latitude'.format(title):'latitude'}, inplace = True) 
        df.rename(columns = {'{}_longitude'.format(title):'longitude'}, inplace = True) 
        # Drop invalid datapoints
        indexNames = df[df['latitude'] == 0].index
        df.drop(indexNames , inplace=True)
        indexNames = df[df['latitude'] < BBox[2]].index
        df.drop(indexNames , inplace=True)
        indexNames = df[df['latitude'] > BBox[3]].index
        df.drop(indexNames , inplace=True)
        indexNames = df[df['longitude'] <BBox[0]].index
        df.drop(indexNames , inplace=True)
        indexNames = df[df['longitude'] >BBox[1]].index
        df.drop(indexNames , inplace=True)
        df_cor=df.loc[:, ['pickup_date','pickup_time','latitude','longitude']]
            
        year = self.year
        month = time.strptime(self.month,'%b').tm_mon
        day = self.day
        hour = self.hour_start
        minutes = self.min_start
        string = "{}-{:0=2d}-{}, {:0=2d}:{:0=2d}:00".format(year, month, day, hour, minutes)
        if "pickup" in file: 
            print("*"*40)
            print(string)
            print("*"*40)
        
        # Filter the data within that day
        self.string = "{}-{:0=2d}-{}".format(year, month, day)
        df_cor = df_cor[df_cor['pickup_date'] == self.string]
        # Filter the data within the hour range
        endtime = self.endtime - timedelta(seconds=1)
        
        df_cor['pickup_time'] = pd.to_datetime(df_cor['pickup_time'])
        df_cor = (df_cor.set_index('pickup_time')
                .between_time(self.starttime.strftime('%H:%M:%S'), endtime.strftime('%H:%M:%S'))
                .reset_index()
                .reindex(columns=df.columns))
        
        return df_cor

    def return_all_result(self): 
        """ ["saved_time_pickup", "saved_rides_pickup", "saved_time_dropoff", "saved_rides_dropoff", 
        "av_per_saved_time_pickup", "av_per_saved_rides_pickup", "av_per_saved_time_dropoff", "av_per_saved_rides_dropoff",
        "av_computation_time_pickup", "av_computation_time_dropoff"]"""
        result = self.rides_total
        for dic_saved in self.saved_totals: 
            for key, value in dic_saved.items(): 
                result.append(value) 
        return result
  

def main():
    # Pick a pool window and show result for that day
    # year = "2016"
    # month = "may"
    # day = "20"
    # hour_start = 17
    # min_start = 30
    # pool_size = 5
    df_center_cor, full_time_matrix = PreloadData().return_data()
    while True: 
        con = input("Continue ? (y or n)")
        if con == "y": 
            year = input("Pick a year (for example: 2016): ")
            month = input("Pick a month (for example: may):")
            day = input("Pick a day (for example: 20): ")
            hour_start = int(input("Pick a hour (for example: 17): "))
            min_start = int(input("Pick a minute (for example: 30): "))
            pool_size = int(input("Pick a pool size (for example: 5): "))
            savefig = True
            PlotAWindow(year=year, month=month, day=day, hour_start=hour_start, min_start=min_start, savefig = savefig, 
                        pool_size=pool_size, df_center_cor = df_center_cor, full_time_matrix = full_time_matrix, batch=False)
            print("Showing the graphs...")
        else: 
            break


if __name__ == '__main__':
    main()  
