We should have two sections, thermal and power generation. Thermal should contain coating modelling and heater modelling. Power generation should include solar panel optimisation. We could have a .csv file bridge the gape between these two. i.e. have the thermal code generate a .csv file, then populate predetermined cells, then have the power generation section read the important numbers from those cells and use those values. Run thermal modelling first, then run the power generation model. I think we should unconstrain the dimensions of the body of the satellite, it feels like its too specific a use case and we aren't optimising anything.

We can purge the github and re-add all necessary code.

A .csv also gives us a much more readable output than the code window on your screen.
