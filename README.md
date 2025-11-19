**Introduction**
Our code handles the thermal modelling of a satellite in GEO orbit. Please run ... first, it will ask for some input values. There is a limit on what values you can enter to prevent unrealistic modelling. Please allow some time for the code to run, it has a lot of calculating to do! 

... will output a report in your terminal with values such as optimal heater power, energy used per year by each coating at this power etc. It will then create an .xlsx file in the same folder as the code which will contain some data and two graphs (search for satellite_analysis_"date_time".xlsx). **THIS FILE IS IMPORTANT FOR THE NEXT SECTION OF OUR CODE**. The two graphs show the six-month evolution of temperature of the satellite with the heater on and they show a 30 day window around the start of the eclipse. 

The optimal heater power is of particular importance to the next section of our code.

**TO RUN OUR SECOND CODE YOU NEED TO CLOSE THE EXCEL FILE.** 

Once you have run the first part of our code, ..., please close it and open our next code .... This code will produce a solar panel area thats required for the satellite to function. This code will take into account the lifetime efficiency of the satellite, internal energy generation and heater power. It will produce a graph showing lifetime efficiency, it will produce a table showing lfetime efficiency degradation at its max, mid and min and it will produce a table showing important values. It will then write these values and these graphs into the excel report for you to use and interpret. 
