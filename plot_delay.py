import os
import time
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

class DelayTime: 
    def __init__(self):
        fig, ax = plt.subplots()
        x = [0, 10, 20]
        y1 = [20.81, 28.79, 32.60]    # 5min, from LGA
        y2 = [22.77, 31.91, 35.47]    # 5min, to LGA
        y3 = [26.40, 32.68, 34.69]    # 10min, from LGA
        y4 = [29.26, 36.26, 37.58]    # 10min, to LGA
        y5 = [21.43, 32.51, 41.33]    # 5min, from LGA
        y6 = [22.94, 34.65, 41.81]    # 5min, to LGA
        y7 = [27.13, 36.79, 44.12]    # 10min, from LGA
        y8 = [29.41, 39.25, 44.32]    # 10min, to LGA
        ax.plot(x, y1, label = "5min pool, from LGA")
        ax.plot(x, y2, label = "5min pool, to LGA")
        ax.plot(x, y3, label = "10min pool, from LGA")
        ax.plot(x, y4, label = "10min pool, to LGA")

        ax.set(xlabel='Percentage of delay allowed (%)', ylabel='Average percentage of distance saved per pool (%)')
        ax.grid()
        plt.legend()
        plt.xticks(x)

        fig.savefig("Result/delay_distance.png", dpi = 300)
        # plt.show()


class DelayRides: 
    def __init__(self):
        fig, ax = plt.subplots()
        x = [0, 10, 20]
        y1 = [20.81, 28.79, 32.60]    # 5min, from LGA
        y2 = [22.77, 31.91, 35.47]    # 5min, to LGA
        y3 = [26.40, 32.68, 34.69]    # 10min, from LGA
        y4 = [29.26, 36.26, 37.58]    # 10min, to LGA
        y5 = [21.43, 32.51, 41.33]    # 5min, from LGA
        y6 = [22.94, 34.65, 41.81]    # 5min, to LGA
        y7 = [27.13, 36.79, 44.12]    # 10min, from LGA
        y8 = [29.41, 39.25, 44.32]    # 10min, to LGA
        ax.plot(x, y5, label = "5min pool, from LGA")
        ax.plot(x, y6, label = "5min pool, to LGA")
        ax.plot(x, y7, label = "10min pool, from LGA")
        ax.plot(x, y8, label = "10min pool, to LGA")

        ax.set(xlabel='Percentage of delay allowed (%)', ylabel='Average percentage of rides saved per pool (%)')
        ax.grid()
        plt.legend()
        plt.xticks(x)

        fig.savefig("Result/delay_rides.png", dpi = 300)
        # plt.show()
        

def main():
    DelayTime()
    DelayRides()


if __name__ == '__main__':
    main()  
