import numpy as np

def SampleGrid(Setup):
    # Create x, y, and z-values for the grid
    x_values = (np.arange(Setup['Observation']['xSamples']) * Setup['Observation']['xSamplingDistance'] - 
                (Setup['Observation']['xSamples'] - 1) * Setup['Observation']['xSamplingDistance'] / 2 + 
                Setup['Observation']['Center'][0])
                
    y_values = (np.arange(Setup['Observation']['ySamples']) * Setup['Observation']['ySamplingDistance'] - 
                (Setup['Observation']['ySamples'] - 1) * Setup['Observation']['ySamplingDistance'] / 2 + 
                Setup['Observation']['Center'][1])
                
    z_values = (np.arange(Setup['Observation']['zSamples']) * Setup['Observation']['zSamplingDistance'] - 
                (Setup['Observation']['zSamples'] - 1) * Setup['Observation']['zSamplingDistance'] / 2 + 
                Setup['Observation']['Center'][2])

    # Create a list to store the points
    points = []

    # Create a sub-struct for each microphone containing the microphone x, y, z-coordinates
    for z in z_values:
        for y in y_values:
            for x in x_values:
                points.append([x, y, z])

    Setup['Observation']['Point'] = points

    
