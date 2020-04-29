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
        y1 = [21.14, 27.84, 30.89]    # 5min, from LGA
        y2 = [22.87, 27.81, 30.66]    # 5min, to LGA
        y3 = [26.87, 32.64, 33.45]    # 10min, from LGA
        y4 = [29.45, 32.45, 33.92]    # 10min, to LGA
        y5 = [22.09, 30.96, 40.06]    # 5min, from LGA
        y6 = [23.17, 29.40, 37.29]    # 5min, to LGA
        y7 = [27.90, 36.07, 43.39]    # 10min, from LGA
        y8 = [29.65, 35.19, 41.18]    # 10min, to LGA
        ax.plot(x, y1, label = "5min pool, from LGA")
        ax.plot(x, y2, label = "5min pool, to LGA")
        ax.plot(x, y3, label = "10min pool, from LGA")
        ax.plot(x, y4, label = "10min pool, to LGA")

        ax.set(xlabel='Percentage of delay allowed (%)', ylabel='Average percentage of distance saved per pool (%)')
        ax.grid()
        plt.legend()

        fig.savefig("Result/delay_time.png", dpi = 300)
        plt.xticks(x)
        # plt.show()


class DelayRides: 
    def __init__(self):
        fig, ax = plt.subplots()
        x = [0, 10, 20]
        y1 = [21.14, 27.84, 30.89]    # 5min, from LGA
        y2 = [22.87, 27.81, 30.66]    # 5min, to LGA
        y3 = [26.87, 32.64, 33.45]    # 10min, from LGA
        y4 = [29.45, 32.45, 33.92]    # 10min, to LGA
        y5 = [22.09, 30.96, 40.06]    # 5min, from LGA
        y6 = [23.17, 29.40, 37.29]    # 5min, to LGA
        y7 = [27.90, 36.07, 43.39]    # 10min, from LGA
        y8 = [29.65, 35.19, 41.18]    # 10min, to LGA
        ax.plot(x, y5, label = "5min pool, from LGA")
        ax.plot(x, y6, label = "5min pool, to LGA")
        ax.plot(x, y7, label = "10min pool, from LGA")
        ax.plot(x, y8, label = "10min pool, to LGA")

        ax.set(xlabel='Percentage of delay allowed (%)', ylabel='Average percentage of rides saved per pool (%)')
        ax.grid()
        plt.legend()

        fig.savefig("Result/delay_rides.png", dpi = 300)
        plt.xticks(x)
        # plt.show()
        

def main():
    DelayTime()
    DelayRides()


if __name__ == '__main__':
    main()  
