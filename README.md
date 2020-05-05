# CS 581: DBMS Ride Sharing Project

### Instructor: [Prof. Ouri Wolfson](https://www.cs.uic.edu/k-teacher/ouri-wolfsonphd)

### Requirements
  - Python 3.5+ 
  - GraphHopper Directions API
  - Pandas python package
  - Numpy python package
  - Matplotlib python package
  - CSV package for python
  
### GraphHopper Directions API Installation & Steps to Setup (Only needed for precomputation)
  - Unzip **graphhopper-web-0.6.0-bin.zip** file provided into the folder where python scripts will be executed 
  - Download & paste the [[**new-york-latest.osm.pbf**](https://uofi.box.com/s/83drgoxa6coomhrnh1phbi5rcps6z57r)] file in the same folder
  - Open command line in this folder
  - Run the following command: **java -jar graphhopper-web-0.6.0-with-dep.jar jetty.resourcebase=webapp config=config-example.properties osmreader.osm=new-york.osm.pbf**
  - It may take a couple of minutes to set up the server
  - You can view the server at "http://localhost:8989


### How to run the code
Below is the list of steps you can do. You can skip certain steps assuming you already have result files from those steps. Before running any code, please download all data files from Data folder [[here](https://uofi.box.com/s/e32xj3oerls3bsmrbkju855iqc1jbqi1)]. 

#### 1. Generate Ride sharing centers
Run Generate_centers.py, you will get a csv containing all ride sharing cluster centers. Also the code will automatically plot a New York map with all ride sharnig centers. Note: Please add Lagruadia as the first center to the csv file. 

For anyone wants to skip this step, check the result file "center_points.csv" in the root folder. 

#### 2. Precompute the distance matrix between centers
Run Precompute.py, you will get a csv file containing the ditance matrix between all ride sharing centers generated in step 1. In order to run this file, first you need to correctly setup Graphhopper local API, which is explaind above. The program will create and overwrite a csv file called "time_matrix.csv". Please notice that this step takes a very long time, approximate running time is 40+hrs. 

For anyone wants to skip this step, check the result file [["time_matrix.csv"](https://uofi.box.com/s/zabtnaz90jsp700l2detxe7f7ksp2rb6)] shared on Box. **You need this file to proceed**. Once you download it, please put it in the root folder.  

#### 3. Compute ride sharing results

Run ride_sharing.py. There are two ways to run this code: 

Uncomment line 497, comment line 506: Pick a day, hour range and the list of pool sizes, return the result for that time period. A few plots will be generated. 

Comment line 497, uncomment line 506: Do a batch calculation. Parameters can be selected by changing line 500-503. This can be done for a whole year (Jul 2015-Jun 2016), for a month or for a day. Parameter "reporttime" takes either days or hours, meaning whether you want to track daily statistics or hourly statistics. The result will be saved to a csv file, with the name including the details about the time period you picked. Depends on the "reporttime" you picked, the file will end with "days" or "hours". 

#### 4. Visualize the result from step 3
First make sure the file you generated in step 3 has a name matching either line 15, 111 or line 208 in result_visualization.py. If not please change the file name in those lines. If your result file end with "hours", please uncomment line 264, comment line 265 and 266. If your result file end with "days", please comment line 264, uncomment line 265 and 266. After everything is set up, you can run result_visualization.py and the result plots will save to the "Result" folder. (Create a Result folder if it does not exists in the repository. )

#### 5. Play with delay tolerance
You can change the delay in your calculate in step 3 by changing line 302 in ride_shairng.py. By default it's 20% delay. 

#### 6. Plot results versus delays
If you have multiple results from step 3 generated with different delays using step 5, you can plot the results versus delays. Here we manually typed in the result for delay 0, 10% and 20%. And running file plot_delay.py will generate a few plots in the Result folder. 

#### 7. Pick a pool window and do instant visualization (Ride sharing Graph)
Run ride_sharing_plot.py. For example, if you pick 2016, may, 01, hour=17, min=30, pool_size=5, then it will do a ride sharing calculation for time window 2016-5-1 17:30-17:35. 


