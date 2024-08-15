import matplotlib.pyplot as plt

def draw_setup(setup):
    """
    Draw the top view of the simulated room with loudspeaker and microphone positions.
    
    Parameters:
    - setup: Dictionary containing the configuration and parameters
    """
    # Initialize figure
    plt.figure()

    # Draw the walls
    x_range = [0, setup['Room']['Dim'][0]]
    y_range = [0, setup['Room']['Dim'][1]]
    
    plt.plot(x_range, [y_range[0], y_range[0]], 'k', linewidth=2)  # Bottom wall
    plt.plot(x_range, [y_range[1], y_range[1]], 'k', linewidth=2)  # Top wall
    plt.plot([x_range[0], x_range[0]], y_range, 'k', linewidth=2)  # Left wall
    plt.plot([x_range[1], x_range[1]], y_range, 'k', linewidth=2)  # Right wall
    
    plt.ylabel('Length [m]')
    plt.xlabel('Width [m]')
    plt.xlim(x_range)
    plt.ylim(y_range)
    plt.gca().set_aspect('equal', adjustable='box')
    
    # Draw Source positions
    for pos in setup['Source']['Position']:
        plt.plot(pos[0], pos[1], 'ro', linewidth=2)
    
    # Draw Microphone positions
    for point in setup['Observation']['Point']:
        plt.plot(point[0], point[1], 'b.', linewidth=2)
    
    plt.show()

# Example usage
if __name__ == "__main__":
    setup = {
        'Room': {'Dim': [10, 8, 3]},
        'Source': {'Position': [[2, 3], [7, 5]]},
        'Observation': {'Point': [[4, 4], [6, 6]]}
    }
    draw_setup(setup)
