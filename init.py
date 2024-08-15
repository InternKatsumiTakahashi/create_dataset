import numpy as np
from SampleGrid import SampleGrid

def create_setup(seed=0, equidistant=False):
    Setup = {}

    # ===========================
    #       Setup
    # ---------------------------
    # Sampling frequency [Hz]
    Setup['Fs'] = 1200

    # Impulse response length [s]
    Setup['Duration'] = 1

    # ===========================
    #       Ambient
    # ---------------------------
    # Ambient temperature [deg C]
    Setup['Ambient'] = {}
    Setup['Ambient']['Temp'] = 20

    # Ambient pressure [Pa]
    Setup['Ambient']['Pressure'] = 1000e2

    # Specific heat capacity of dry air, sea level, 0 deg C
    # Isobaric molar heat capacity [J/mol deg K]
    cp = 29.07
    # Isochore molar heat capacity [J/mol deg K]
    cv = 20.7643

    # Ratio of specific heats
    gamma = cp / cv

    # Ideal gas constant [J/kg deg K]
    R = 287

    # Density of air [kg/m3]
    Setup['Ambient']['rho'] = Setup['Ambient']['Pressure'] / (R * (Setup['Ambient']['Temp'] + 273.15))

    # Speed of sound [m/s]
    Setup['Ambient']['c'] = np.sqrt(gamma * Setup['Ambient']['Pressure'] / Setup['Ambient']['rho'])

    # ===========================
    #       Room
    # ---------------------------
    # Room dimensions [x, y, z] [m]
    z = 2.4
    x = 0
    y = 0
    rng = np.random.default_rng(seed=seed)
    while 20 > x * y or 60 < x * y:
        x = 2.83 + (4.87 - 2.83) * rng.random()
        y = 1.1 * x + (4.5 * x - 9.6 - 1.1 * x) * rng.random()

    Setup['Room'] = {}
    Setup['Room']['Dim'] = [x, y, z]
    Setup['Room']['ReverbTime'] = 0.6
    Setup['Room']['Dim2'] = x * y

    # ===========================
    #       Source Array
    # ---------------------------
    # Source lower cutoff frequency
    Setup['Source'] = {}
    Setup['Source']['Highpass'] = 10  # [Hz]

    # Source higher cutoff frequency
    Setup['Source']['Lowpass'] = 500  # [Hz]

    # Source position [x, y, z] [m]
    sx = Setup['Room']['Dim'][0] * rng.random()
    sy = Setup['Room']['Dim'][1] * rng.random()
    sz = 0
    Setup['Source']['Position'] = [[sx, sy, sz]]
    Setup['Source']['SrcNum'] = len(Setup['Source']['Position'])

    # ===========================
    #       Observation region
    # ---------------------------
    # Noise in microphones
    Setup['Observation'] = {}
    Setup['Observation']['NoiseLevel'] = 0  # [dB SPL]

    # Observation point x, y, z-position [m]
    Setup['Observation']['xSamples'] = 32
    Setup['Observation']['ySamples'] = 32
    Setup['Observation']['zSamples'] = 1
    Setup['Observation']['xSamplingDistance'] = Setup['Room']['Dim'][0] / (Setup['Observation']['xSamples'] - 1)  # [m]
    Setup['Observation']['ySamplingDistance'] = Setup['Room']['Dim'][1] / (Setup['Observation']['ySamples'] - 1)  # [m]
    Setup['Observation']['zSamplingDistance'] = 1  # [m]

    if equidistant:
        Setup['Observation']['xSamples'] = int(Setup['Room']['Dim'][0] * 10) + 2
        Setup['Observation']['ySamples'] = int(Setup['Room']['Dim'][1] * 10) + 2
        Setup['Observation']['zSamples'] = 1
        Setup['Observation']['xSamplingDistance'] = 0.1
        Setup['Observation']['ySamplingDistance'] = 0.1

    # Center of microphone array [x, y, z] [m]
    Setup['Observation']['Center'] = [Setup['Room']['Dim'][0] / 2, Setup['Room']['Dim'][1] / 2, 0]

    # Call a function to handle the sample grid, you will need to define SampleGrid in Python
    SampleGrid(Setup)
    return Setup

# Call the init function
if __name__ == "__main__":
    from pprint import pprint
    pprint(create_setup())
