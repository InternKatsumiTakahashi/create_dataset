import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from init import create_setup
from green_3d_freq_modal_response import green_3d_freq_modal_response, green_3d_freq_modal_response_z0
from draw_setup import draw_setup
import pickle

def create_dataset(num_rooms,plot=0,save=True):

    savedata = {}

    for j in range(1, num_rooms + 1):
        print(f"Room {j}")
        Setup = create_setup(seed=j, equidistant=0)

        # Generate observed data
        freq_lim = 400
        psi_r, mu, psi_s = green_3d_freq_modal_response_z0(freq_lim,Setup)

        frequency_response = np.zeros((len(psi_r), len(mu[0]), len(psi_s[0])))
        for i in range(len(mu[0])):
            frequency_response[:, i, :] = np.dot(psi_r, np.diag(mu[:, i])) @ psi_s

        # Reshape to 3D arrays
        x_coor = np.arange(0, Setup['Room']['Dim'][0] + Setup['Observation']['xSamplingDistance'], Setup['Observation']['xSamplingDistance'])
        y_coor = np.arange(0, Setup['Room']['Dim'][1] + Setup['Observation']['ySamplingDistance'], Setup['Observation']['ySamplingDistance'])
        frequency = np.arange(0, Setup['Fs']/2, 1/Setup['Duration'])

        frequency_response = frequency_response.squeeze()
        # frequency_responseの1次元目の引数について... (2次元目は周波数領域)
        # 0,1,...,len(x_coor)-1 は隣接
        # 0,1*len(x_coor),2*len(x_coor),...,(len(y_coor)-1)*len(x_coor) は隣接

        frequency_response = frequency_response.reshape(len(y_coor), len(x_coor), len(frequency))
        abs_frequency_response = np.abs(frequency_response)

        savedata[f'Room{j+1}'] = {'FrequencyResponse': frequency_response.reshape(-1,len(frequency)),
                                  'edges': {},
                                  'Setup': Setup}

        if plot:
            # Draw the simulated setup
            draw_setup(Setup)

            # Plot the Transfer function at a given frequency
            freq_idces = [100,300,500]

            for freq_idx in freq_idces:
                plt.figure()
                #plt.contourf(y_coor, x_coor, abs_frequency_response[:, :, freq_idx].T, edgecolor='none')
                plt.imshow(frequency_response[:,:,freq_idx])
                plt.xlabel('X-dimension [m]')
                plt.ylabel('Y-dimension [m]')
                plt.title(f'Contour plot of TF magnitude throughout the room at f = {frequency[freq_idx]:.1f} Hz')
                plt.colorbar()
                plt.show()
        
        if save:
            with open('tensors.pkl', 'wb') as f:
                pickle.dump(savedata, f)


if __name__ == "__main__":
    create_dataset(100,plot=0)
