# CS581 Database Management System Ride Sharing Project

import os
import time
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests


class PickADay:
    """Pick a day and hour range from a day, calculate ride sharing result for all pools with different pool sizes. """
    
    def __init__(self, year, month, day, hour_start, hour_end, pool_size, df_center_cor = None, full_time_matrix = None, plot=False, batch=False):
        self.total_start_time = time.time()
        self.year = year
        self.month = month
        self.day = day
        self.hour_start = hour_start
        self.hour_end= hour_end
        self.string = ""
        self.batch = batch
        
        file1 = "Data/{}_pickup_{}.csv".format(month, year)
        file2 = "Data/{}_dropoff_{}.csv".format(month, year)
        self.df_cor1 = self.readdatafile(file1, "dropoff")
        self.df_cor2 = self.readdatafile(file2, "pickup")
        self.rides_total = [len(self.df_cor1), len(self.df_cor2)]
        self.df_titles = ["pickup", "dropoff"]
        self.dfs = [self.df_cor1, self.df_cor2]
        if df_center_cor is not None: 
            self.df_center_cor = df_center_cor
        else: 
            self.df_center_cor = pd.read_csv("center_points.csv")
        self.list_of_centers = []
        for i in range(len(self.df_center_cor)): 
            row = self.df_center_cor.iloc[i]
            cor = [row["longitude"], row["latitude"]]
            self.list_of_centers.append(cor)
        # These are the limit ranges to find the closest center to each ride share request: 
        self.longlimit = 0.005
        self.latlimit = 0.005
        # Load the time matrix with transportation time between every center pairs: 
        if full_time_matrix is not None: 
            self.full_time_matrix = full_time_matrix
        else: 
            self.full_time_matrix = np.genfromtxt('time_matrix.csv', delimiter=',')
        # The pool window size we consider: 
        self.cut_windows = pool_size
        
        self.saved_totals = []  # Each item in the list contains the result for one pool size. The items in the list are dictionaries, which are defined below. 
        for window in self.cut_windows: 
            dic_saved = {"saved_time_pickup":0, "saved_rides_pickup":0, "saved_time_dropoff":0, "saved_rides_dropoff":0, 
                         "av_per_saved_time_pickup":0, "av_per_saved_rides_pickup":0, "av_per_saved_time_dropoff":0, "av_per_saved_rides_dropoff":0,
                         "av_computation_time_pickup":0, "av_computation_time_dropoff":0}
            self.saved_totals.append(dic_saved)
        
        self.run()
        if not self.batch: 
            self.final_report()
        else:
            self.total_end_time = time.time()
            print("Time Taken to run everything for this day: {:.2f}s".format(self.total_end_time-self.total_start_time))
        if plot: 
            self.plotall()
        
    def run(self):    
        for ind_window, window in enumerate(self.cut_windows):  # For each time window(pool size): 
            print("Processing for time window (pool size): {}mins".format(window))
            # Wihtout ride sharing, what the time and rides would be: 
            time_total, rides_total = [0, 0], [0, 0]
            
            num_windows = int(60*(self.hour_end-self.hour_start)/window)
            for df_index, df in enumerate(self.dfs): # One for pickup one for dropoff
                start_time = time.time()
                starttime = datetime.strptime('{}:00:00'.format(self.hour_start), '%H:%M:%S')
                for i in range(num_windows): 
                    # First, get the coordinate matrix and time matrix for all requests in this time window: 
                    endtime, df_cor = self.get_all_data_in_window(df, starttime, window)
                    starttime = endtime   # Set the new starttime for the next time window 
                    df_time = self.cal_time_matrix(df_cor)
                    # Run algorithm
                    saved_time, saved_rides = RideSharingAlgorithm(df_cor, df_time, algorithm=0).return_report()
                    # Calculate the total time and total rides without save: 
                    for t in df_time[0]: 
                        time_total[df_index] += t
                    rides_total[df_index] += len(df_cor) 
                    # Calculate maga data
                    self.saved_totals[ind_window]["saved_time_{}".format(self.df_titles[df_index])] += saved_time
                    self.saved_totals[ind_window]["saved_rides_{}".format(self.df_titles[df_index])] += saved_rides
                    if not self.batch: 
                        print("For {} at the airport: Total time saved: {:.1f}min. Total number of trips saved: {}. "\
                            .format(self.df_titles[df_index], self.saved_totals[ind_window]["saved_time_{}".format(self.df_titles[df_index])], \
                                self.saved_totals[ind_window]["saved_rides_{}".format(self.df_titles[df_index])]))
                        print("Pool processed ({}): {}/{}".format(self.df_titles[df_index], i, num_windows))
                end_time = time.time()
                average_time = (end_time-start_time)/num_windows
                
                # In case there is no request in this pool, assign 1 to time and rides. 
                if time_total[df_index] < 0.01: 
                    time_total[df_index] = 1
                if rides_total[df_index] < 0.1: 
                    rides_total[df_index] = 1
                        
                self.saved_totals[ind_window]["av_computation_time_{}".format(self.df_titles[df_index])] = average_time
                self.saved_totals[ind_window]["av_per_saved_time_{}".format(self.df_titles[df_index])] = \
                    self.saved_totals[ind_window]["saved_time_{}".format(self.df_titles[df_index])] / time_total[df_index]
                self.saved_totals[ind_window]["av_per_saved_rides_{}".format(self.df_titles[df_index])] = \
                    self.saved_totals[ind_window]["saved_rides_{}".format(self.df_titles[df_index])] / rides_total[df_index]
            print("Processing for time window (pool size): {}mins is done. ".format(window))

    def final_report(self):
        print("~"*60)
        print("Final report: ")
        print("Test Time: {} {}-{}. ".format(self.string, '{:0=2d}:00:00'.format(self.hour_start), '{:0=2d}:00:00'.format(self.hour_end)))
        print("Total trips (from LGA): {}; Total trips (to LGA): {}. ".format(self.rides_total[0], self.rides_total[1]))
        for ind_window, window in enumerate(self.cut_windows):
            print("For time window (pool size): {}mins: ".format(window))
            print("Total number of pools: {}".format(int(60*(self.hour_end-self.hour_start)/window)))
            print("For pickup at the airport: Total time saved: {:.1f}min. Total number of trips saved: {}. "\
                .format(self.saved_totals[ind_window]["saved_time_pickup"], self.saved_totals[ind_window]["saved_rides_pickup"]))
            print("For dropoff at the airport: Total time saved: {:.1f}min. Total number of trips saved: {}. "\
                .format(self.saved_totals[ind_window]["saved_time_dropoff"], self.saved_totals[ind_window]["saved_rides_dropoff"]))
            print("For pickup at the airport: Average time saved per pool (%): {:.1f}. Average number of trips saved per pool (%): {:.1f}. "\
                .format(self.saved_totals[ind_window]["av_per_saved_time_pickup"]*100, self.saved_totals[ind_window]["av_per_saved_rides_pickup"]*100))
            print("For dropoff at the airport: Average time saved per pool (%): {:.1f}. Average number of trips saved per pool (%): {:.1f}. "\
                .format(self.saved_totals[ind_window]["av_per_saved_time_dropoff"]*100, self.saved_totals[ind_window]["av_per_saved_rides_dropoff"]*100))
            print("Average running time per pool: For pickup: {:.2f}s. For dropoff: {:.2f}s. "\
                .format(self.saved_totals[ind_window]["av_computation_time_pickup"], self.saved_totals[ind_window]["av_computation_time_dropoff"]))
            print("*"*60)
        self.total_end_time = time.time()
        print("Time Taken to run everything: {:.2f}s".format(self.total_end_time-self.total_start_time))
        print("~"*60)
        # print(self.saved_totals)
    
    def plotall(self): 
        y1 = [self.saved_totals[0]["av_per_saved_time_pickup"]*100, self.saved_totals[1]["av_per_saved_time_pickup"]*100]
        y2 = [self.saved_totals[0]["av_per_saved_time_dropoff"]*100, self.saved_totals[1]["av_per_saved_time_dropoff"]*100]
        y_label = "Average distance saved per pool (%)"
        self.plot_result(y1, y2, y_label, "1")
        y1 = [self.saved_totals[0]["av_per_saved_rides_pickup"]*100, self.saved_totals[1]["av_per_saved_rides_pickup"]*100]
        y2 = [self.saved_totals[0]["av_per_saved_rides_dropoff"]*100, self.saved_totals[1]["av_per_saved_rides_dropoff"]*100]
        y_label = "Average number of trips saved per pool (%)"
        self.plot_result(y1, y2, y_label, "2", auto_y_limit=False)
        y1 = [self.saved_totals[0]["av_computation_time_pickup"], self.saved_totals[1]["av_computation_time_pickup"]]
        y2 = [self.saved_totals[0]["av_computation_time_dropoff"], self.saved_totals[1]["av_computation_time_dropoff"]]
        y_label = "Average computation time per pool (s)"
        self.plot_result(y1, y2, y_label, "3")
    
    def plot_result(self, y1, y2, y_label, plotname, auto_y_limit=True): 
        plt.figure(figsize=(9,6)) 
        x = self.cut_windows.copy()
        x.extend(x)
        plt.plot(self.cut_windows, y1, marker = "o", linewidth=3, label = "From LGA")
        plt.plot(self.cut_windows, y2, marker = "o", linewidth=3, label = "To LGA")
        y1.extend(y2)
        for x,y in zip(x, y1):
            label = "{:.2f}".format(y)
            plt.annotate(label, # this is the text
                        (x,y), # this is the point to label
                        textcoords="offset points", # how to position the text
                        xytext=(0,10), # distance from text to points (x,y)
                        ha='center') # horizontal alignment can be left, right or center
        plt.xticks(self.cut_windows)
        if auto_y_limit:
            plt.ylim([min(y1)*0.8, max(y1)*1.2])
        plt.xlabel('Pool Times (min)')
        plt.ylabel(y_label)
        plt.legend()  
        plt.savefig('{}.png'.format(plotname), dpi=300)   
        plt.show()
    
    def get_all_data_in_window(self, df, starttime, timewindow):
        """ Given the dataframe, the start time of the timewindow, and the size of the timewindow, 
        return the endtime and a new filtered dataframe with lag and long"""
        
        timewindow = timedelta(minutes=timewindow)
        endtime = starttime + timewindow
        endtime_1 = endtime - timedelta(seconds=1)  # To remove the right boundary from "between_time" method of df. 
        
        df = (df.set_index('pickup_time')
                .between_time(starttime.strftime('%H:%M:%S'), endtime_1.strftime('%H:%M:%S'))
                .reset_index()
                .reindex(columns=df.columns))
        df = df[['latitude','longitude']]   # Only keep the coordinates of all requests in a time window. 
        
        return endtime, df

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
        string = "{}-{:0=2d}-{}, {:0=2d}:00:00".format(year, month, day, hour)
        if "pickup" in file: 
            print("*"*40)
            print(string)
            print("*"*40)
        
        # Filter the data within that day
        self.string = "{}-{:0=2d}-{}".format(year, month, day)
        df_cor = df_cor[df_cor['pickup_date'] == self.string]
        # Filter the data within the hour range
        starthour = datetime.strptime('{}:00:00'.format(self.hour_start), '%H:%M:%S')
        if self.hour_end == 24: # datatime raise error for time "24:00:00"
            endhour = datetime.strptime('00:00:00', '%H:%M:%S')
        else: 
            endhour = datetime.strptime('{}:00:00'.format(self.hour_end), '%H:%M:%S')
        endhour_1 = endhour - timedelta(seconds=1)  # To remove the right boundary from "between_time" method of df. 
        df_cor['pickup_time'] = pd.to_datetime(df_cor['pickup_time'])
        df_cor = (df_cor.set_index('pickup_time')
                .between_time(starthour.strftime('%H:%M:%S'), endhour_1.strftime('%H:%M:%S'))
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
               
    
class RideSharingAlgorithm:
    """Apply constraints and find the ride sharing plan (rsp)"""
    
    def __init__(self, df_cor, df_time, algorithm=0):
        self.df_cor, self.df_time = df_cor, df_time
        self.num_data = len(self.df_cor)
        self.laguardia = [40.7769, -73.874]
        
        self.delays = 0.0 # Tolerable delay percentage
        self.possible_shares = {}   # Key is a string. For example, "001002001" means a possible ride sharing between 001 and 002, with going to 001 first. Value is the time saved. 
        
        self.apply_constraints()
        if algorithm == 0: 
            self.search()
        elif algorithm == 1: 
            self.stable_matching()
        elif algorithm == 2: 
            self.k_cluster()

        # self.plot_possible_shares()
        # self.plot_sharing_results()
        
    def apply_constraints(self):
        for i in range(self.num_data):
            for j in range(i+1, self.num_data): # "d" in the variables below stand for destination/origin, in this case is the Laguardia airport
                t_da = self.df_time[0, i+1]
                t_db = self.df_time[0, j+1]
                t_ab = self.df_time[i+1, j+1]
                t_ba = self.df_time[j+1, i+1]
                
                if t_da <= t_ba and t_db <= t_ab:  # First constraint
                    pass
                elif (t_da + t_ab) > (t_db*(1+self.delays)) and (t_db + t_ba) > (t_da*(1+self.delays)):   # Second constraint
                    pass
                else: # When a ride shaing is possible. Notice that if both going to A and B first works, we'll go to the closet one first. 
                    id1 = "{0:0=3d}{1:0=3d}{2:0=3d}".format(i, j, i)
                    id2 = "{0:0=3d}{1:0=3d}{2:0=3d}".format(i, j, j)
                    
                    if (t_da + t_ab) < (t_db*(1+self.delays)) and (t_db + t_ba) < (t_da*(1+self.delays)):
                        if t_da <= t_db: 
                            self.possible_shares[id1] = t_db - t_ab
                        else: 
                            self.possible_shares[id2] = t_da - t_ba
                    elif (t_da + t_ab) < (t_db*(1+self.delays)): # Can only go to A first
                        self.possible_shares[id1] = t_db - t_ab
                    else: # Can only go to B first
                        self.possible_shares[id2] = t_da - t_ba
        # print("All possible ride sharings: ")
        # print(self.possible_shares)
        # print(len(self.possible_shares))
    
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
    
    def plot_possible_shares(self, title="pickup", folder="", show = True, save=False): 
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
                    ax.plot(x, y, marker = 'o', markersize=0, linewidth=0.5, color='orange')
        # Plot all share request points
        ax.scatter(self.df_cor.longitude, self.df_cor.latitude, zorder=1, alpha= 1, c='b', s=10)
        # Plot Laguardia airport as single red dot
        ax.scatter(self.laguardia[1], self.laguardia[0], alpha= 1, c='r', marker = 'x', s=80)
        if title == "pickup": 
            title2 = "from"
        else: 
            title2 = "to"
        ax.set_title('All possible ride sharing pairs ({} LGA)'.format(title2))
        ax.set_xlim(BBox[0],BBox[1])
        ax.set_ylim(BBox[2],BBox[3])
        ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= ratio*1.8)
        if show: 
            plt.show()
        if save: 
            fig.savefig(folder + os.sep + 'plot_possible_shares_{}.png'.format(title), dpi=300)
        
    def plot_sharing_results(self, title="pickup", folder="", show = True, save=False): 
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
        ax.scatter(self.laguardia[1], self.laguardia[0], alpha= 1, c='r', marker = 'x', s=80)
        if title == "pickup": 
            title2 = "from"
        else: 
            title2 = "to"
        ax.set_title('Ridesharing Graph ({} LGA)'.format(title2))
        ax.set_xlim(BBox[0],BBox[1])
        ax.set_ylim(BBox[2],BBox[3])
        ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= ratio*1.8)
        if show: 
            plt.show()
        if save: 
            fig.savefig(folder + os.sep + 'plot_sharing_results_{}.png'.format(title), dpi=300)   
    
    def return_report(self):
        return self.total_saved_max, len(self.shared_rides_max)


class Batch: 
    def __init__(self, start, end, poolsize, reporttime = "days"):
        self.start =  start
        self.end = end
        self.start_time = datetime.strptime(self.start, '%Y-%m-%d')
        self.end_time = datetime.strptime(self.end, '%Y-%m-%d')
        self.loop_time = self.start_time
        self.poolsize = poolsize
        self.reporttime = reporttime
        
        self.df_center_cor = pd.read_csv("center_points.csv")
        self.full_time_matrix = np.genfromtxt('time_matrix.csv', delimiter=',')
        
        self.results = []
        
        while self.loop_time <= self.end_time: 
            if reporttime == "days":
                hour_start = 0
                hour_end = 24
            else: 
                hour_start = self.loop_time.hour
                hour_end = hour_start + 1
                
            # print("{}-{}-{}".format(str(self.loop_time.year), self.loop_time.strftime("%b").lower(), '{:0=2d}'.format(self.loop_time.day)))
            result = PickADay(year=str(self.loop_time.year), month=self.loop_time.strftime("%b").lower(), day='{:0=2d}'.format(self.loop_time.day), \
                                hour_start=hour_start, hour_end=hour_end, pool_size=self.poolsize, \
                                df_center_cor = self.df_center_cor, full_time_matrix = self.full_time_matrix, plot=False, batch=True).return_all_result()
            self.results.append([self.loop_time.strftime('%Y-%m-%d'), self.loop_time.strftime('%H:%M:%S')]+result)
            if reporttime == "days":
                self.loop_time += timedelta(days=1)
            else: 
                self.loop_time += timedelta(hours=1)
        print("*"*40)
        print("Finished batch calculation for all dates. ")
        #Save the result to file
        self.save_result()
    
    def save_result(self):
        columns = ["Date", "Time", "Total_trips_from_LGA", "Total_trips_to_LGA"]
        contents = ["saved_time_pickup", "saved_rides_pickup", "saved_time_dropoff", "saved_rides_dropoff", 
            "av_per_saved_time_pickup", "av_per_saved_rides_pickup", "av_per_saved_time_dropoff", "av_per_saved_rides_dropoff",
            "av_computation_time_pickup", "av_computation_time_dropoff"]
        for pool in self.poolsize: 
            for content in contents: 
                columns.append("{}min_".format(pool)+content)
        df = pd.DataFrame(self.results, columns=columns)
        if self.reporttime != "days": 
            df.drop(df.tail(1).index,inplace=True)  # Drop last row if reporttime is in hours
        filename = 'results_{}_to_{}_in_{}.csv'.format(self.start, self.end, self.reporttime)
        df.to_csv(filename, float_format='%.4f', index=False)
        print("Result saved to {}. \n".format(filename))
        
def main():
    # Pick a day and show result for that day
    # PickADay(year="2016", month="may", day="13", hour_start=12, hour_end=24, pool_size=[5, 10], plot=True, batch=False)
    
    # Pick a range of days, calculate the result and save to csv file. This probably only need to run one or a few times. 
    start = '2015-07-01'
    end = '2016-06-30'
    poolsize = [5, 10]
    reporttime = "days" # days or hours. 
    # If in hours, the "end" parameter above should be set to the next day of the desired end date. 
    
    Batch(start, end, poolsize, reporttime=reporttime)

if __name__ == '__main__':
    main()       