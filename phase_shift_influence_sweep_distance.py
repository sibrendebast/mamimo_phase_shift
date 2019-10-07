from math import e, pi, sqrt, floor
import numpy as np
import matplotlib.pyplot as plt

max_num_sub = 2
freq = 2.61E9
BW = 20E6
BW_step = 20E6
max_BW = 1E9
c = 2.998E9

num_antennas = 64
user_distance = 0.1
distance_step = 0.1
max_distance = 5

num_users = 2

antenna_spacing = 0.07  # meter
user_spacing = 0.5  # meter

# locations_antenna = [(i*antenna_spacing, 0) for i in range(num_antennas)]
# receiver_locations = [(user_spacing*i, 1) for i in range(num_users)]

power = []


def dist(p1, p2):
    return sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


results = []
results_dB = []
distances = []
num_subs = [1, 2, 1200]
while user_distance <= max_distance:
    locations_antenna = [(i*antenna_spacing - num_antennas/2*antenna_spacing + antenna_spacing/2, 0) for i in range(num_antennas)]
    receiver_locations = [(user_spacing*i - num_users/2*user_spacing + user_spacing/2, user_distance) for i in range(num_users)]

    distance = []
    for ant in locations_antenna:
        distance.append([])
        for rec in receiver_locations:
            distance[-1].append(dist(rec, ant))

    # print(distance)

    delta_dist = []
    for i in range(len(distance)):
        delta_dist.append([])
        for j in range(len(distance[i])):
            delta_dist[i].append(distance[i][j] - distance[i][0])

    print("User distance:", user_distance)
    power = []
    power_dB = []
    for num_sub in num_subs:
        # print('number of subcarriers:', num_sub)
        if num_sub == 1:
            sub_freq = [freq]
        else:
            subspace = BW/(num_sub-1)
            sub_freq = [freq-BW/2 + subspace*g for g in range(num_sub)]
        # print(sub_freq)
        delta_phi = np.zeros((num_antennas, num_users, num_sub), dtype=np.cdouble)
        for h in range(len(delta_dist)):
            for j in range(len(delta_dist[h])):
                for k in range(int(num_sub)):
                    delta_phi[h, j, k] = 2*pi*sub_freq[k]*delta_dist[h][j]/c
        # print(delta_phi)

        signal_vectors = np.zeros((num_users), dtype=np.cdouble)
        for g in range(len(delta_phi)):
            for h in range(len(delta_phi[g])):
                for k in range(len(delta_phi[g][h])):
                    # power_sub = (1-abs(freq-sub_freq[k])/BW)
                    # print(power_sub)
                    signal_vectors[h] += e**(delta_phi[g, h, k]*1j)

        # print(signal_vectors)
        norm_sig = []
        norm_sig_dB = []
        for sig in signal_vectors:
            norm_sig.append(abs(sig/signal_vectors[0]))
            norm_sig_dB.append(10*np.log10(abs(sig/signal_vectors[0])))
        # print(norm_sig[1])

        power.append(norm_sig[1])
        power_dB.append(norm_sig_dB[1])

    results.append(power)
    results_dB.append(power_dB)
    distances.append(user_distance)
    user_distance += distance_step

results = np.array(results).T
results_dB = np.array(results_dB).T
print(results.shape)
# normalise
norm_results = []
norm_results_dB = []
print(results)
for idx in range(results.shape[1]):
    norm_results.append(results[:, idx]/results[:, 0])
    norm_results_dB.append(10*np.log10(results[:, idx]/results[:, 0]))

norm_results = np.array(norm_results).T
norm_results_dB = np.array(norm_results_dB).T


plt.figure()
# ax = plt.subplots(111)
plt.title("Impact of multi-sine waveform \nin function of the distance to the BS")
labels = np.array(["1 Subcarrier", "2 Subcarriers", "1024 Subcarriers"])
for idx, l in enumerate(labels):
    plt.plot(distances, norm_results_dB[idx], label=l)
plt.xlabel('User distance from the basestation [m]')
# plt.xticks(num_subs)
plt.ylabel('Normalised Amplitude Gain [dB]')
plt.grid()
plt.legend()
plt.savefig('gain_vs_distance_dB.eps', bbox_inches='tight', pad_inches=0)
plt.savefig('gain_vs_distance_dB.png', bbox_inches='tight', pad_inches=0)
# plt.show()

plt.figure()
# ax = plt.subplots(111)
plt.title("Impact of multi-sine waveform \nin function of the distance to the BS")
labels = np.array(["1 Subcarrier", "2 Subcarriers", "1024 Subcarriers"])
for idx, l in enumerate(labels):
    plt.plot(distances, norm_results[idx], label=l)
plt.xlabel('User distance from the basestation [m]')
# plt.xticks(num_subs)
plt.ylabel('Normalised Amplitude Channel Gain')
plt.grid()
plt.legend()
plt.savefig('gain_vs_distance.eps', bbox_inches='tight', pad_inches=0)
plt.savefig('gain_vs_distance.png', bbox_inches='tight', pad_inches=0)
