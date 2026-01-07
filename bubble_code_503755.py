# -- coding: utf-8 --
"""
Created on Tue Jun  3 18:44:22 2025
@author: parai
"""
import re
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Shared folder for all MSD plots
MSD_FOLDER = "all_msd_analysis"  # Change this to your preferred name
Path(MSD_FOLDER).mkdir(exist_ok=True)

data_file = (r"C:/Users/parai/Downloads/wetfoam2-bub-RH2-0-140000-0-503755.DAT.txt")
num_steps = 2087

def parse_bubble_data(filename):
    """
    Parse bubble tracking data file and extract x,y coordinates for each bubble across time steps.

    Args:
        filename (str): Path to the data file

    Returns:
        tuple: (bubble_x_coords, bubble_y_coords, time_steps)
            - bubble_x_coords: dict with bubble_id as key, list of x coordinates as value
            - bubble_y_coords: dict with bubble_id as key, list of y coordinates as value  
            - time_steps: list of time values for each frame
    """
    
    bubble_x_coords = defaultdict(list)
    bubble_y_coords = defaultdict(list)
    time_steps = []

    current_time = None

    with open(filename, 'r') as file:
        for line in file:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Parse time information from header lines
            if line.startswith('#') and 'time' in line:
                # Extract time value using regex
                time_match = re.search(r'time\s=\s(\d+)', line)
                if time_match:
                    current_time = int(time_match.group(1))
                    time_steps.append(current_time)
                continue

            # Skip other header/comment lines
            if line.startswith('#'):
                continue

            # Parse data lines (format: id x y area pressure Z)
            parts = line.split()
            if len(parts) >= 6:
                try:
                    bubble_id = int(parts[0])
                    x_coord = float(parts[1])
                    y_coord = float(parts[2])

                    bubble_x_coords[bubble_id].append(x_coord)
                    bubble_y_coords[bubble_id].append(y_coord)

                except (ValueError, IndexError):
                    # Skip lines that don't match expected format
                    continue

    return dict(bubble_x_coords), dict(bubble_y_coords), time_steps

x_coords,y_coords,times = parse_bubble_data(data_file)

def unwrap_bubble_trajectories(x_coords, y_coords):  
    x_unwrapped = {}
    y_unwrapped = {}

    for bubble_id in x_coords.keys():
        x_unwrapped[bubble_id] = np.unwrap(np.array(x_coords[bubble_id]), period=21.566555)
        y_unwrapped[bubble_id] = np.unwrap(np.array(y_coords[bubble_id]), period=21.566555)

    # Return dictionaries, not numpy arrays
    return x_unwrapped, y_unwrapped

x_unwrapped, y_unwrapped = unwrap_bubble_trajectories(x_coords,y_coords)

def plot_bubble_trajectory(bubble_id):
    if bubble_id in x_coords:
        lifetime = len(x_coords[bubble_id])
        plt.plot(x_coords[bubble_id],y_coords[bubble_id],label = 'Lifetime = ' + str(lifetime), marker = ',')
        #plt.legend()

def plot_unwrapped_trajectory(bubble_id):
       if bubble_id in x_unwrapped:
           lifetime = len(x_unwrapped[bubble_id])
           plt.plot(x_unwrapped[bubble_id],y_unwrapped[bubble_id],label = 'Lifetime = ' + str(lifetime), marker = ',')
           #plt.legend()   

def plot_n_raw(n):  
    for i in range(1,n+1):   
        plot_bubble_trajectory(i)   

def plot_n_unwrapped(n):           
    for i in range(1,n+1):
        plot_unwrapped_trajectory(i)
        plt.legend()
        plt.show()
        
def mean_squared_displacement(bubble_id):
    if bubble_id not in x_unwrapped:
        return None, None
    
    x_traj = np.array(x_unwrapped[bubble_id])
    y_traj = np.array(y_unwrapped[bubble_id])
    
    dx = np.diff(x_traj)
    dy = np.diff(y_traj)
    
    msd = np.cumsum(dx**2 + dy**2)
    
    steps = np.arange(1, len(msd) + 1)
    
    return msd, steps

def individual_msd(bubble_id):
    msd, steps = mean_squared_displacement(bubble_id)
    #lifetime = len(x_unwrapped[bubble_id])
    #plt.plot(msd, steps, label = lifetime)
    #plt.legend()
    return msd, steps

def total_msd():
    
    all_msds = np.zeros(num_steps - 1)
    steps = np.arange(1,num_steps)
    
    available_bubbles = list(x_unwrapped.keys())
    
    for bubble_id in available_bubbles:
        
        msd = individual_msd(bubble_id)[0]
        all_msds[:len(msd)] += msd
    
    all_msds  /= num_steps
    return all_msds, steps


def theory_plot(n, alpha,C):
    theory_steps = C*np.arange(1,n+1)
    theory_points = theory_steps * (2/alpha)
    #plt.plot(theory_points,theory_steps)
   
    return theory_points, theory_steps

theory_points, theory_steps = theory_plot(num_steps-1,1,10)
all_msds, steps = total_msd()


plt.figure(figsize=(12, 10), dpi=300)   
plt.plot(steps,all_msds,label = 503755)
#plt.plot(theory_steps,theory_points)
plt.title('Plot of MSD vs Steps')
plt.legend()
plt.xlabel('Steps')
plt.ylabel('Mean Square Displacement(Simulation Units)')
ax = plt.gca()
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
#ax.spines['right'].set_color('none')
#ax.spines['top'].set_color('none')
plt.grid()
plt.savefig('MSD503755.pdf')
plt.show()

plt.figure(figsize=(12, 10), dpi=300)   
plt.loglog(steps,all_msds,label = 503755)
plt.title('Log-Log Plot of MSD vs Steps')
plt.grid()
plt.legend()
plt.xlabel('Steps')
plt.ylabel('Mean Square Displacement(Simulation Units)')
plt.savefig('loglogMSD503755.pdf')
plt.show()






#Plot all Bubble Trajectories
'''
plt.figure(figsize=(12, 10), dpi=300)      
plot_n_raw(400)
plt.tight_layout()
plt.title('Raw Bubble Trajectories')
plt.show() 
'''
'''
plt.figure(figsize=(12, 10), dpi=300)
plot_n_unwrapped(400)   
plt.tight_layout()
#plt.title('Unwrapped Bubble Trajectories')
#plt.show() 
'''