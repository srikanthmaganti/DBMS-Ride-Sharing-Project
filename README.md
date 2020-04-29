# CS 581: DBMS Ride Sharing Project

### Instructor: [Prof. Ouri Wolfson](https://www.cs.uic.edu/k-teacher/ouri-wolfsonphd)

### Requirements
  - Python 3.5+ 
  - MySQL Workbench 6.3
  - MySQL connector for python
  - GraphHopper Directions API
  - Pandas python package
  - Numpy python package
  - Matplotlib python package
  - CSV package for python
  - New York City Map (in .pbf format) 
  
### GraphHopper Directions API Installation & Steps to Setup
  - Unzip **graphhopper-web-0.6.0-bin.zip** file provided into the folder where python scripts will be executed 
  - Download & paste the **new-york-latest.osm.pbf** file in the same folder
  - Open command line in this folder
  - Run the following command: **java -jar graphhopper-web-0.6.0-with-dep.jar jetty.resourcebase=webapp config=config-example.properties osmreader.osm=new-york.osm.pbf**
  - It may take a couple of minutes to set up the server
  - You can view the server at "http://localhost:8989

