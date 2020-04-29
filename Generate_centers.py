import datetime
import time
import json
import matplotlib.pyplot as plt
import mysql.connector
from mysql.connector import Error
import numpy as np
import pandas as pd
import requests
import csv
from mysql.connector import errorcode
from urllib.request import URLError, Request, urlopen
from datetime import datetime, timedelta


class ReadDataFile:
    def __init__(self, file, type):
        BBox = [-74.2587, -73.6860, 40.4929, 40.9171]
        
        df = pd.read_csv(file)
        df.rename(columns = {'{}_latitude'.format(type):'latitude'}, inplace = True) 
        df.rename(columns = {'{}_longitude'.format(type):'longitude'}, inplace = True) 
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
        self.df_cor=df.loc[:, ['latitude','longitude']]
    
    def return_df_cor(self):
        return self.df_cor
    
    
class GetCenters:
    """ Compute all cluster centers"""
    
    def __init__(self, df_cor):
        self.df_cor = df_cor
        self.laguardia = [40.7769, -73.874]
        self.longlimit = 0.002
        self.latlimit = 0.002
        self.num_data = len(self.df_cor)
        
        self.list_of_centers = []
        # print(self.df_cor.head(20))
        
        self.create_centers()
        self.plot_centers()
        self.save_centers()   # Uncomment this line to save the centers to file. 
        print(len(self.list_of_centers))
        
    
    def create_centers(self): 
        for i in range(self.num_data): 
            if i % 100 == 0:
                print("{:.2f}% Completed".format(i/self.num_data*100))
            data = self.df_cor.iloc[i]
            cor = [data["longitude"], data["latitude"]]
            if_center = self.check_if_belong_to_center(cor)
            if not if_center: 
                self.list_of_centers.append(cor)   
    
    def plot_centers(self):
        BBox = ((-74.2587, -73.6860, 40.4929, 40.9171))
        # BBox = ((-74.2387, -73.6860, 40.4929, 40.9171))
        ruh_m = plt.imread('New_York_City_District_Map_wikimedia.png')
        ratio = (BBox[3]-BBox[2])/(BBox[1]-BBox[0])
        fig, ax = plt.subplots(figsize = (8,8*ratio))
        
        # Draw the radius box
        ax.scatter([i[0] for i in self.list_of_centers], [i[1] for i in self.list_of_centers], zorder=1, alpha= 1, c='b', s=10)
        
        # Plot Laguardia airport as single red dot
        ax.scatter(self.laguardia[1], self.laguardia[0], alpha= 1, c='r', s=20)
        
        # Set global parameters
        ax.set_title('All Ride Sharing Centers')
        ax.set_xlim(BBox[0],BBox[1])
        ax.set_ylim(BBox[2],BBox[3])
        ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= ratio*1.8)
        plt.show()
        # fig.savefig('{}.png'.format("1"), dpi=300)              

    def save_centers(self):
        df = pd.DataFrame(self.list_of_centers, columns=["longitude", "latitude"])
        df.to_csv('center_points.csv', index=False)
    
    def check_if_belong_to_center(self, cor): 
        for center in self.list_of_centers: 
            if abs(center[0]-cor[0]) <= self.longlimit and abs(center[1]-cor[1]) <= self.latlimit: 
                return True
        return False


class CheckRadius:
    """ Check if a certain radius limit is reasonable. This class is only for reference purpose"""
    
    def __init__(self): 
        self.laguardia = [40.7769, -73.874]
        self.longlimit = 0.002
        self.latlimit = 0.002
        self.plot_points()
    
    def plot_points(self):
        BBox = ((-74.2587, -73.6860, 40.4929, 40.9171))
        # BBox = ((-74.2387, -73.6860, 40.4929, 40.9171))
        ruh_m = plt.imread('New_York_City_District_Map_wikimedia.png')
        ratio = (BBox[3]-BBox[2])/(BBox[1]-BBox[0])
        fig, ax = plt.subplots(figsize = (8,8*ratio))
        
        # Draw the radius box
        ax.scatter(-73.98, 40.78, zorder=1, alpha= 1, c='b', s=10)
        ax.axvline(x= -73.98-self.longlimit)
        ax.axvline(x= -73.98+self.longlimit)
        ax.axhline(y= 40.78-self.latlimit)
        ax.axhline(y= 40.78+self.latlimit)
        
        # Plot Laguardia airport as single red dot
        ax.scatter(self.laguardia[1], self.laguardia[0], alpha= 1, c='r', s=20)
        
        # Set global parameters
        ax.set_title('Raiuds Test')
        ax.set_xlim(BBox[0],BBox[1])
        ax.set_ylim(BBox[2],BBox[3])
        ax.imshow(ruh_m, zorder=0, extent = BBox, aspect= ratio*1.8)
        plt.show()
        # fig.savefig('{}.png'.format("1"), dpi=300)
        
            
def main():
    # CheckRadius()
    
    df_cor1 = ReadDataFile("Data/may_pickup_2016.csv", "dropoff").return_df_cor()
    df_cor2 = ReadDataFile("Data/may_dropoff_2016.csv", "pickup").return_df_cor()
    frames = [df_cor1, df_cor2]
    df_cor = pd.concat(frames)
    GetCenters(df_cor)


if __name__ == '__main__':
    main()
    
