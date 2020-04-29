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

  
class Precompute:
    def __init__(self):
        self.laguardia = [40.7769, -73.874]
        self.df_cor = pd.read_csv("center_points.csv")
        
        self.num_rows = len(self.df_cor)
        self.num_cols = len(self.df_cor)
        
        self.distance_matrix = np.zeros((self.num_rows, self.num_cols))
        # self.find_matrix()
        self.compute()
    
    def compute(self): 
        for i in range(0, self.num_rows): 
            print(i)
            if i % 10 == 0:
                np.savetxt("time_matrix.csv", self.distance_matrix, delimiter=",", fmt='%.1f')
            for j in range(self.num_cols): 
                if i == j: 
                    continue
                data1 = self.df_cor.iloc[i]
                cor1 = [data1["longitude"], data1["latitude"]]
                data2 = self.df_cor.iloc[j]
                cor2 = [data2["longitude"], data2["latitude"]]
                result = self.find_distance(cor1[1], cor1[0], cor2[1], cor2[0])
                self.distance_matrix[i, j] = result
        np.savetxt("time_matrix.csv", self.distance_matrix, delimiter=",", fmt='%.1f')
        
    def distance_miles(self, distance):
        miles = distance / 1609.344
        return round(miles, 2)

    def time_mins(self, time):
        minute, reminder = divmod(time, 60000)
        if reminder >= 50:
            minute += 1
        return minute
        
    def find_distance(self, pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude):
        request_str = 'http://localhost:8989/route?point=' + str(pickup_latitude) + '%2C' + str(pickup_longitude) + '&point=' + str(
            dropoff_latitude) + '%2C' + str(dropoff_longitude) + '&vehicle=car'
        request = Request(request_str)

        try:
            response = urlopen(request)
            req = requests.get(request_str)
            output = json.loads(req.text)
            paths = output["paths"]
            # distance = self.distance_miles(paths[0]["distance"])
            time = float(self.time_mins(paths[0]["time"]))
            # result = [distance, time]
            result = time
        except URLError:
            # result = [-1, -1]
            result = -1

        return result

    def find_matrix(self): 
        self.url = "https://graphhopper.com/api/1/matrix"
        lat_long=self.df_cor.values.tolist()
        lat_long=[self.laguardia]+lat_long
        lat_long_list = ["{}, {}".format(i[0], i[1]) for i in lat_long][:10]
        # print(lat_long_list)
        Dict = {"type":"json","vehicle":"car","debug":"true","out_array":["weights","times","distances"],"key":"0bad3151-9a5a-4838-91ad-bce60e86fa0a"}
        Dict.update( {"point" : lat_long_list} )
        # print(Dict.get("point"))
        while True: 
            try: 
                response = requests.request("GET", self.url, params=Dict)

                parsed = json.loads(response.text)
                self.df_time = pd.DataFrame(parsed["times"])
                self.df_time.to_csv('time_matrix.csv', index=False)
                # print(json.dumps(parsed, indent=4, sort_keys=True))
                break
            except KeyError: 
                print("Timeout. Retrying...")
                time.sleep(3)
                 
            
def main(): 
    Precompute()


if __name__ == '__main__':
    main()
    
