from nasaCEA import Engine
from channelSizing import ChannelSizing
from channelSizing import Channel
from channelSizing import Geometry
import matplotlib.pyplot as plt
import numpy as np

engineObj = Engine() # creates the engine
engineObj.CEAoutput()
engineObj.getEngineProps() # generates CEA values

# setting geometry values
channel_range = np.linspace(80, 80, 1) # N
height_range = np.linspace(0.0001, 0.01, 10) # h  
width_range = np.linspace(0.00258, 0.00258, 1) # w, need to set constraints

channels = ChannelSizing(engineObj, N=channel_range, h=height_range, w=width_range) 
channels.axial_solver() # thermo calcs for heat flux and channel geometry

height_geom = []
flux_y = []

for station in range(30,0,-1): 
    best_geometry = [channels.channel_props[station][0].num_channels, channels.channel_props[station][0].height, channels.channel_props[station][0].width]
    height_geom.append(channels.channel_props[station][0].height)
    
for station in range(30,0,-1):
    flux_y.append(channels.channel_props[station][1].flux)
    thermo = [channels.channel_props[station][1].q_RP, channels.channel_props[station][1].q_gas, channels.channel_props[station][1].q_wall, channels.channel_props[station][1].flux]
    
print(height_geom)
print(flux_y)



# PLOTTING FOR PRESENTATION IMAGES

plt.plot(engineObj.engineProps[::-1, 0], flux_y)
plt.savefig('heatFlux6.png')
plt.pause(10)
plt.close()

plt.plot(engineObj.engineProps[::-1, 0], height_geom)
plt.savefig('height_vs_Z_new')
plt.pause(10)
plt.close()


x = engineObj.engineProps[:,0]
y = engineObj.engineProps[:,4]

# plt.plot(x, y)
# plt.xlabel("Distance from Injector (m)")
# plt.ylabel("Mach Number")
# plt.title("CEA Combustion Gas Mach Number") 
# plt.savefig('Mach_vs_Z')
# plt.show()

    