import numpy as np
from scipy.signal import butter, filtfilt

def green_3d_freq_modal_response_z0(freq_lim, setup):
    """
    Calculate the modal response in the frequency domain for a lightly damped rectangular room.
    
    Parameters:
    - freq_lim: Highest eigenfunction resonance frequency included in the calculations
    - setup: Dictionary containing the configuration and parameters
    
    Returns:
    - Psi_s: Eigenfunctions evaluated at source positions (size = [nMod, sPos])
    - Psi_r: Eigenfunctions evaluated at receiver positions (size = [rPos, nMod])
    - Mu: Eigenvalues of the eigenfunctions at each excitation frequency (size = [nMod, nFreq])
    """
    
    # Initialize parameters for the room
    V = np.prod(setup['Room']['Dim'])
    A_xy = np.prod(setup['Room']['Dim'][:2])
    A_yz = np.prod(setup['Room']['Dim'][1:])
    A_xz = np.prod([setup['Room']['Dim'][0], setup['Room']['Dim'][2]])
    S = 2 * (A_xy + A_yz + A_xz)
    
    # Absorption coefficient
    alpha = 24 * np.log(10) / setup['Ambient']['c'] * V / (S * setup['Room']['ReverbTime'])
    beta = alpha / 8
    
    # Time constants for different mode types
    tau_oblique = V / (2 * setup['Ambient']['c'] * S * beta)
    tau_tangential = 3 * V / (5 * setup['Ambient']['c'] * S * beta)
    tau_axial = 3 * V / (4 * setup['Ambient']['c'] * S * beta)
    tau_compression = V / (setup['Ambient']['c'] * beta) * 1 / S
    
    # Determine solution frequencies
    frequency = np.arange(0, setup['Fs'] / 2, 1 / setup['Duration'])
    w = 2 * np.pi * frequency
    
    # Low frequency rolloff of driver
    B, A = butter(2, 2 * setup['Source']['Highpass'] / setup['Fs'], 'high')
    imp = np.concatenate(([1], np.zeros(len(w) - 1)))
    imp = filtfilt(B, A, imp)
    
    # High frequency rolloff / anti-aliasing filter
    B, A = butter(2, 2 * setup['Source']['Lowpass'] / setup['Fs'])
    imp = filtfilt(B, A, imp)
    freq_win = np.fft.fft(imp, 2 * len(w))
    freq_win = freq_win[:len(w)]
    
    # Extract coordinates
    points = np.array([p for p in setup['Observation']['Point']])
    sources = np.array([s for s in setup['Source']['Position']])
    
    x, y = points[:, 0], points[:, 1]
    x_s, y_s = sources[:, 0], sources[:, 1]
    k = w / setup['Ambient']['c']
    
    # Frequency limit for modes
    min_dim = np.min(setup['Room']['Dim'][:2])
    max_modal_number = np.ceil(2 * freq_lim * min_dim / setup['Ambient']['c'])
    
    # Create list of all possible combinations of modal numbers
    modal_numbers = pick_n(np.arange(max_modal_number + 1), 2, 'all')
    
    # Calculate resonance frequencies corresponding to the list of modal numbers
    res_freqs = setup['Ambient']['c'] / (2 * np.pi) * np.sqrt(
        np.sum((np.pi * modal_numbers / setup['Room']['Dim'][:2]) ** 2, axis=1)
    )
    
    # Sort modes according to resonance frequency
    sorted_idx = np.argsort(res_freqs)
    res_freqs = res_freqs[sorted_idx]
    modal_numbers = modal_numbers[sorted_idx, :]
    
    # Prune the list of modes to only include modes with resonance frequencies below freq_lim
    valid_indices = res_freqs < freq_lim
    modal_numbers = modal_numbers[valid_indices, :]
    km = 2 * np.pi * res_freqs[valid_indices] / setup['Ambient']['c']
    
    # Calculate responses
    psi_s = np.zeros((len(km), len(setup['Source']['Position'])))
    psi_r = np.zeros((len(setup['Observation']['Point']), len(km)))
    mu = np.zeros((len(km), len(w)))
    
    for mode_index in range(len(modal_numbers)):
        eps = np.count_nonzero(modal_numbers[mode_index] > 0)
        eps = 2 ** eps if eps > 0 else 1
        
        psi_r[:, mode_index] = np.sqrt(eps / V) * (
            np.cos(modal_numbers[mode_index, 0] * np.pi * x / setup['Room']['Dim'][0]) *
            np.cos(modal_numbers[mode_index, 1] * np.pi * y / setup['Room']['Dim'][1])
        )
        psi_s[mode_index, :] = np.sqrt(eps / V) * (
            np.cos(modal_numbers[mode_index, 0] * np.pi * x_s / setup['Room']['Dim'][0]) *
            np.cos(modal_numbers[mode_index, 1] * np.pi * y_s / setup['Room']['Dim'][1])
        )
        
        if eps == 0:
            taum = tau_compression
        elif len(modal_numbers[mode_index]) == 1:
            taum = tau_axial
        elif len(modal_numbers[mode_index]) == 2:
            taum = tau_tangential
        else:
            raise ValueError('Invalid modal dimension. Should be between 0 and 2.')
        
        mu[mode_index, :] = -4 * np.pi / (k ** 2 - km[mode_index] ** 2 - 1j * k / (taum * setup['Ambient']['c'])) * freq_win
    
    mu[:, 0] = 0  # Hardcode DC-component to zero
    
    return psi_r, mu, psi_s

# Example usage
if __name__ == "__main__":
    setup = {
        'Room': {'Dim': [10, 8, 3], 'ReverbTime': 0.6},
        'Ambient': {'c': 343},
        'Source': {'Highpass': 10, 'Lowpass': 500, 'Position': [[1, 2]]},
        'Observation': {'Point': [[4, 5]]},
        'Fs': 1200,
        'Duration': 1
    }
    freq_lim = 1000
    psi_r, mu, psi_s = green_3d_freq_modal_response_z0(freq_lim, setup)
    print('Psi_r:', psi_r)
    print('Mu:', mu)
    print('Psi_s:', psi_s)
