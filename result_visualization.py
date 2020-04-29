import os
import time
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = [10, 5]
from datetime import datetime, timedelta
import requests
from pathlib import Path


class Hourly:
    def __init__(self, save=False):
        self.rawdata = pd.read_csv("results_2016-05-01_to_2016-06-01_in_hours.csv")
        
        parameters = ["saved_time_pickup", "saved_rides_pickup", "saved_time_dropoff", "saved_rides_dropoff", 
        "av_per_saved_time_pickup", "av_per_saved_rides_pickup", "av_per_saved_time_dropoff", "av_per_saved_rides_dropoff",
        "av_computation_time_pickup", "av_computation_time_dropoff"]
        parameters_names = ["Total_saved_time_from_LGA", "Total_saved_rides_from_LGA", "Total_saved_time_to_LGA", "Total_saved_rides_to_LGA",
                            "Average_percent_of_time_saved_per_pool_from_LGA", "Average_percent_of_rides_saved_per_pool_from_LGA", 
                            "Average_percent_of_time_saved_per_pool_to_LGA", "Average_percent_of_rides_saved_per_pool_to_LGA", 
                            "Average_computation_time_per_pool_from_LGA", "Average_computation_time_per_pool_to_LGA"]
        columns = ["Total_trips_from_LGA", "Total_trips_to_LGA"]
        # Generate the name of all columns
        for window in [5, 10]: 
            for para in parameters: 
                columns.append("{}min_{}".format(window, para))
        self.df = self.rawdata.groupby('Time', as_index=False)[columns].mean()
        self.df['Time'] = self.df['Time'].apply(lambda x: x[:2])
        self.df.rename(columns = {'Time':'Hour of day', "Total_trips_from_LGA": "Average_Daily_Trips_from_LGA", "Total_trips_to_LGA": "Average_Daily_Trips_to_LGA"}, inplace = True) 
        dic_names = {}
        # Change the name of all columns to more plain english
        for window in [5, 10]: 
            for i, para in enumerate(parameters): 
                dic_names["{}min_".format(window)+para] = parameters_names[i] + "_({}min_pool)".format(window)
        self.df.rename(columns = dic_names, inplace = True) 
        # Multipli all percentage columns by 100
        for window in [5, 10]: 
            for _, para in enumerate(parameters_names[:-2]):
                self.df[para + "_({}min_pool)".format(window)] = self.df[para + "_({}min_pool)".format(window)] * 100
        # print(self.df)
        
        self.plot0(save)
        self.plot1(save)
        self.plot2(save)
        self.plot3(save)
    
    def plot0(self, save): 
        ax = plt.gca()
        self.df.plot(kind='bar',x='Hour of day',y=["Average_Daily_Trips_from_LGA", "Average_Daily_Trips_to_LGA"], color=['blue', 'red'], ax=ax)
        ax.set_ylabel("Average number of trips per hour")
        if save: 
            plt.savefig('Result/Hourly_rides.png', dpi=300)
        else: 
            plt.show()
        plt.clf()
        
    def plot1(self, save): 
        ax = plt.gca()
        ax2 = ax.twinx()
        self.df.plot(kind='bar',x='Hour of day',y=["Average_Daily_Trips_from_LGA", "Average_Daily_Trips_to_LGA"], color=['blue', 'red'], ax=ax)
        self.df.plot(kind='line',x='Hour of day',y=["Average_percent_of_time_saved_per_pool_from_LGA_(5min_pool)", "Average_percent_of_time_saved_per_pool_to_LGA_(5min_pool)",
                                            "Average_percent_of_time_saved_per_pool_from_LGA_(10min_pool)", "Average_percent_of_time_saved_per_pool_to_LGA_(10min_pool)"], 
                     color=['blue', 'red', 'orange', 'green'], ax=ax2)
        ax.set_ylabel("Average number of trips per hour")
        ax2.set_ylabel("Average percent of time saved per pool (%)")
        ax.set_xlim(-0.5, 23.5)
        if save: 
            plt.savefig('Result/Hourly_av_time_saved.png', dpi=300)
        else: 
            plt.show()
        plt.clf()
    
    def plot2(self, save): 
        ax = plt.gca()
        ax2 = ax.twinx()
        self.df.plot(kind='bar',x='Hour of day',y=["Average_Daily_Trips_from_LGA", "Average_Daily_Trips_to_LGA"], color=['blue', 'red'], ax=ax)
        self.df.plot(kind='line',x='Hour of day',y=["Average_percent_of_rides_saved_per_pool_from_LGA_(5min_pool)", "Average_percent_of_rides_saved_per_pool_to_LGA_(5min_pool)",
                                            "Average_percent_of_rides_saved_per_pool_from_LGA_(10min_pool)", "Average_percent_of_rides_saved_per_pool_to_LGA_(10min_pool)"], 
                     color=['blue', 'red', 'orange', 'green'], ax=ax2)
        ax.set_ylabel("Average number of trips per hour")
        ax2.set_ylabel("Average percent of rides saved per pool (%)")
        ax.set_xlim(-0.5, 23.5)
        if save: 
            plt.savefig('Result/Hourly_av_rides_saved.png', dpi=300)
        else: 
            plt.show()
        plt.clf()
    
    def plot3(self, save): 
        ax = plt.gca()
        ax2 = ax.twinx()
        self.df.plot(kind='bar',x='Hour of day',y=["Average_Daily_Trips_from_LGA", "Average_Daily_Trips_to_LGA"], color=['blue', 'red'], ax=ax)
        self.df.plot(kind='line',x='Hour of day',y=["Average_computation_time_per_pool_from_LGA_(5min_pool)", "Average_computation_time_per_pool_to_LGA_(5min_pool)", \
                                            "Average_computation_time_per_pool_from_LGA_(10min_pool)", "Average_computation_time_per_pool_to_LGA_(10min_pool)"], 
                     color=['blue', 'red', 'orange', 'green'], ax=ax2)
        ax.set_ylabel("Average number of trips per hour")
        ax2.set_ylabel("Average computation time per pool (s)")
        ax.set_xlim(-0.5, 23.5)
        ax2.set_ylim(0, 0.5)
        if save: 
            plt.savefig('Result/Hourly_av_computation_time.png', dpi=300)
        else: 
            plt.show()
        plt.clf()
        

class Daily:
    def __init__(self, save=False):
        self.rawdata = pd.read_csv("results_2015-07-01_to_2016-06-30_in_days.csv")
        
        parameters = ["saved_time_pickup", "saved_rides_pickup", "saved_time_dropoff", "saved_rides_dropoff", 
        "av_per_saved_time_pickup", "av_per_saved_rides_pickup", "av_per_saved_time_dropoff", "av_per_saved_rides_dropoff",
        "av_computation_time_pickup", "av_computation_time_dropoff"]
        parameters_names = ["Total_saved_time_from_LGA", "Total_saved_rides_from_LGA", "Total_saved_time_to_LGA", "Total_saved_rides_to_LGA",
                            "Average_percent_of_time_saved_per_pool_from_LGA", "Average_percent_of_rides_saved_per_pool_from_LGA", 
                            "Average_percent_of_time_saved_per_pool_to_LGA", "Average_percent_of_rides_saved_per_pool_to_LGA", 
                            "Average_computation_time_per_pool_from_LGA", "Average_computation_time_per_pool_to_LGA"]
        columns = ["Total_trips_from_LGA", "Total_trips_to_LGA"]
        # Generate the name of all columns
        for window in [5, 10]: 
            for para in parameters: 
                columns.append("{}min_{}".format(window, para))
        self.rawdata['Date'] = self.rawdata['Date'].apply(lambda x: x[5:7])
        self.df = self.rawdata.groupby('Date', as_index=False)[columns].mean()
        self.df.rename(columns = {'Date':'Month', "Total_trips_from_LGA": "Average_Daily_Trips_from_LGA", "Total_trips_to_LGA": "Average_Daily_Trips_to_LGA"}, inplace = True) 
        dic_names = {}
        # Change the name of all columns to more plain english
        for window in [5, 10]: 
            for i, para in enumerate(parameters): 
                dic_names["{}min_".format(window)+para] = parameters_names[i] + "_({}min_pool)".format(window)
        self.df.rename(columns = dic_names, inplace = True) 
        # Multipli all percentage columns by 100
        for window in [5, 10]: 
            for _, para in enumerate(parameters_names[:-2]):
                self.df[para + "_({}min_pool)".format(window)] = self.df[para + "_({}min_pool)".format(window)] * 100
        # print(self.df)
        
        self.plot0(save)
        self.plot1(save)
        self.plot2(save)
        self.plot3(save)
    
    def plot0(self, save): 
        ax = plt.gca()
        self.df.plot(kind='bar',x='Month',y=["Average_Daily_Trips_from_LGA", "Average_Daily_Trips_to_LGA"], color=['blue', 'red'], ax=ax)
        ax.set_ylabel("Average number of trips per day")
        if save: 
            plt.savefig('Result/Monthly_rides.png', dpi=300)
        else: 
            plt.show()
        plt.clf()
        
    def plot1(self, save): 
        ax = plt.gca()
        ax2 = ax.twinx()
        
        self.df.plot(kind='bar',x='Month',y=["Average_Daily_Trips_from_LGA", "Average_Daily_Trips_to_LGA"], color=['blue', 'red'], ax=ax)
        self.df.plot(kind='line',x='Month',y=["Average_percent_of_time_saved_per_pool_from_LGA_(5min_pool)", "Average_percent_of_time_saved_per_pool_to_LGA_(5min_pool)",
                                            "Average_percent_of_time_saved_per_pool_from_LGA_(10min_pool)", "Average_percent_of_time_saved_per_pool_to_LGA_(10min_pool)"], 
                     color=['blue', 'red', 'orange', 'green'], ax=ax2)
        ax.set_xlim(-0.5, 11.5)
        ax.set_ylabel("Average number of trips per day")
        ax2.set_ylabel("Average percent of time saved per pool (%)")
        if save: 
            plt.savefig('Result/Monthly_av_time_saved.png', dpi=300)
        else: 
            plt.show()
        plt.clf()
    
    def plot2(self, save): 
        ax = plt.gca()
        ax2 = ax.twinx()
        self.df.plot(kind='bar',x='Month',y=["Average_Daily_Trips_from_LGA", "Average_Daily_Trips_to_LGA"], color=['blue', 'red'], ax=ax)
        self.df.plot(kind='line',x='Month',y=["Average_percent_of_rides_saved_per_pool_from_LGA_(5min_pool)", "Average_percent_of_rides_saved_per_pool_to_LGA_(5min_pool)",
                                            "Average_percent_of_rides_saved_per_pool_from_LGA_(10min_pool)", "Average_percent_of_rides_saved_per_pool_to_LGA_(10min_pool)"], 
                     color=['blue', 'red', 'orange', 'green'], ax=ax2)
        ax.set_xlim(-0.5, 11.5)
        ax.set_ylabel("Average number of trips per day")
        ax2.set_ylabel("Average percent of rides saved per pool (%)")
        if save: 
            plt.savefig('Result/Monthly_av_rides_saved.png', dpi=300)
        else: 
            plt.show()
        plt.clf()
    
    def plot3(self, save): 
        ax = plt.gca()
        ax2 = ax.twinx()
        self.df.plot(kind='bar',x='Month',y=["Average_Daily_Trips_from_LGA", "Average_Daily_Trips_to_LGA"], color=['blue', 'red'], ax=ax)
        self.df.plot(kind='line',x='Month',y=["Average_computation_time_per_pool_from_LGA_(5min_pool)", "Average_computation_time_per_pool_to_LGA_(5min_pool)", \
                                            "Average_computation_time_per_pool_from_LGA_(10min_pool)", "Average_computation_time_per_pool_to_LGA_(10min_pool)"], 
                     color=['blue', 'red', 'orange', 'green'], ax=ax2)
        ax.set_xlim(-0.5, 11.5)
        ax2.set_ylim(0, 0.5)
        ax.set_ylabel("Average number of trips per day")
        ax2.set_ylabel("Average computation time per pool (s)")
        if save:  
            plt.savefig('Result/Monthly_av_computation_time.png', dpi=300)
        else: 
            plt.show()
        plt.clf()
  

class Yearly:
    def __init__(self, save=False):
        self.rawdata = pd.read_csv("results_2015-07-01_to_2016-06-30_in_days.csv")
        self.cut_windows = [5, 10]
        
        parameters = ["saved_time_pickup", "saved_rides_pickup", "saved_time_dropoff", "saved_rides_dropoff", 
        "av_per_saved_time_pickup", "av_per_saved_rides_pickup", "av_per_saved_time_dropoff", "av_per_saved_rides_dropoff",
        "av_computation_time_pickup", "av_computation_time_dropoff"]
        columns = ["Total_trips_from_LGA", "Total_trips_to_LGA"]
        # Generate the name of all columns
        for window in [5, 10]: 
            for para in parameters: 
                columns.append("{}min_{}".format(window, para))
        self.rawdata['Date'] = self.rawdata['Date'].apply(lambda x: x[4])
        self.df = self.rawdata.groupby('Date', as_index=False)[columns].mean()
        
        self.plotall()
    
    def plotall(self): 
        y1 = [self.df.iloc[0]["5min_av_per_saved_time_pickup"]*100, self.df.iloc[0]["10min_av_per_saved_time_pickup"]*100]
        y2 = [self.df.iloc[0]["5min_av_per_saved_time_dropoff"]*100, self.df.iloc[0]["10min_av_per_saved_time_dropoff"]*100]
        y_label = "Average distance saved per pool (%)"
        self.plot_result(y1, y2, y_label, "1_yearly")
        y1 = [self.df.iloc[0]["5min_av_per_saved_rides_pickup"]*100, self.df.iloc[0]["10min_av_per_saved_rides_pickup"]*100]
        y2 = [self.df.iloc[0]["5min_av_per_saved_rides_dropoff"]*100, self.df.iloc[0]["10min_av_per_saved_rides_dropoff"]*100]
        y_label = "Average number of trips saved per pool (%)"
        self.plot_result(y1, y2, y_label, "2_yearly", auto_y_limit=False)
        y1 = [self.df.iloc[0]["5min_av_computation_time_pickup"], self.df.iloc[0]["10min_av_computation_time_pickup"]]
        y2 = [self.df.iloc[0]["5min_av_computation_time_dropoff"], self.df.iloc[0]["10min_av_computation_time_dropoff"]]
        y_label = "Average computation time per pool (s)"
        self.plot_result(y1, y2, y_label, "3_yearly")
    
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


def main():
    # Choose the minimum time unit for the visualization
    # Hourly(save=True)
    Daily(save=True)
    Yearly(save=True)


if __name__ == '__main__':
    main()  

