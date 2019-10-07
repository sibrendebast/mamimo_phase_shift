from math import e, pi, sqrt, floor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

max_num_sub = 10000
freq = 2.61E9
BW = 20E6
c = 2.998E9

num_antennas = 64

num_users = 2
antenna_spacing = 0.07  # meter - distance btween the antenna elements
user_spacing = 0.5  # meter - distance between the users
distance_BS = 2.5  # meter - distance between the users and the BS

# Define teh locations of the antenna elements and the users
# locations_antenna = [(i*antenna_spacing, 0) for i in range(num_antennas)]
# receiver_locations = [(user_spacing*i, distance_BS) for i in range(num_users)]
locations_antenna = [(i*antenna_spacing - num_antennas/2*antenna_spacing + antenna_spacing/2, 0) for i in range(num_antennas)]
receiver_locations = [(user_spacing*i - num_users/2*user_spacing + user_spacing/2, distance_BS) for i in range(num_users)]


# plot the users and the BS antennas
plt.figure()
plt.title("Simulation scenario")
plt.scatter([receiver_locations[0][0]],
            [receiver_locations[0][1]],
            label='Target User')
plt.scatter([receiver_locations[1][0]],
            [receiver_locations[1][1]],
            label='Victim User')
plt.scatter([locations_antenna[i][0] for i in range(len(locations_antenna))],
            [locations_antenna[i][1] for i in range(len(locations_antenna))],
            label='BS antenna')
plt.xlabel("X Position [m]")
plt.ylabel("Y Position [m]")
plt.legend()
plt.grid()
plt.savefig('scenario.eps', bbox_inches='tight', pad_inches=0)


# calcultae the distance between two points
def dist(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


# Canculate the distance between all receivers and antennas
distance = []
for ant in locations_antenna:
    distance.append([])
    for rec in receiver_locations:
        distance[-1].append(dist(rec, ant))


# calculate the Delta distance for all receivens and antennas
delta_dist = []
for i in range(len(distance)):
    delta_dist.append([])
    for j in range(len(distance[i])):
        delta_dist[i].append(distance[i][j] - distance[i][0])


# bookkeeping
i = -1
num_sub = floor(1 + 2**i)
num_subs = []
# power logging
power = []
power_dB = []
# check until the maxiilmal number of subcarriers is reached
while num_sub <= max_num_sub:
    print('number of subcarriers:', num_sub)
    # log the number of subcarriers for plotting of the results
    num_subs.append(num_sub)
    # get the frequencies of the subcarriers
    if i == -1:
        sub_freq = [freq]
    else:
        subspace = BW/(num_sub-1)
        sub_freq = [freq-BW/2 + subspace*g for g in range(num_sub)]
    # calculate all delta phi's
    delta_phi = np.zeros((num_antennas, num_users, num_sub), dtype=np.cdouble)
    for h in range(len(delta_dist)):
        for j in range(len(delta_dist[h])):
            for k in range(int(num_sub)):
                delta_phi[h, j, k] = 2*pi*sub_freq[k]*delta_dist[h][j]/c
    # print(delta_phi)

    # add the signal components together
    signal_vectors = np.zeros((num_users), dtype=np.cdouble)
    for g in range(len(delta_phi)):
        for h in range(len(delta_phi[g])):
            for k in range(len(delta_phi[g][h])):
                signal_vectors[h] += e**(delta_phi[g, h, k]*1j)

    # print(signal_vectors)
    # normalise the results
    norm_sig = []
    norm_sig_dB = []
    for sig in signal_vectors:
        norm_sig.append(abs(sig/signal_vectors[0]))
        norm_sig_dB.append(10*np.log10(abs(sig/signal_vectors[0])))
    print(norm_sig[1])

    power.append(norm_sig[1])
    power_dB.append(norm_sig_dB[1])
    i += 1
    num_sub = 1 + 2**i

# normalise against the defaults Value
norm_power = []
norm_power_dB = []
for pow in power:
    norm_power.append(pow/power[0])
    norm_power_dB.append(10*np.log10(pow/power[0]))

# plot the results using the dB scale and the normal scale
plt.figure()
plt.title("Channel gain due to subcarriers")
plt.semilogx(num_subs, norm_power_dB)
plt.xlabel('Number of subcarriers')
plt.ylabel('Normalised Channel Amplitude Gain [dB]')
plt.grid()
plt.gca().ticklabel_format(axis='y', style='plain', useOffset=False)
plt.savefig('gain_vs_subcarriers_dB.eps', bbox_inches='tight', pad_inches=0)


plt.figure()
plt.title("Impact of multi-sine waveform \nin function of the number of subcarriers")
plt.semilogx(num_subs, norm_power)
plt.xlabel('Number of subcarriers')
plt.ylabel('Normalised Channel Amplitude Gain')
plt.grid()
plt.gca().ticklabel_format(axis='y', style='plain', useOffset=False)
plt.savefig('gain_vs_subcarriers.eps', bbox_inches='tight', pad_inches=0)
