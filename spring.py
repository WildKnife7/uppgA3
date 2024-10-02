import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = "data/rig1_measure1.tsv"
spring_names = ["frame", "time", "x1", "y1", "z1", "x2", "y2", "z2"]

df = pd.read_csv(path, sep="\t", skiprows=11, names=spring_names)

samplerate = 100.0
t = df["time"].to_numpy()

z1 = df["z1"].to_numpy() / 1000.0
z2 = df["z2"].to_numpy() / 1000.0


def fourier_transform(t, x, samplerate):
    fourier = np.fft.rfft(x)
    xfourier = np.fft.rfftfreq(len(t), 1.0/samplerate)

    return (xfourier, fourier)

def index_of_nearest(a, val):
    diff = np.abs(a - val)
    return diff.argmin()


plt.plot(t, z1)
plt.plot(t, z2)
plt.show()

t_cutoff_f = float(input("start time for fourier: "))
t_cutoff_i = index_of_nearest(t, t_cutoff_f)

trimmed_t = t[t_cutoff_i:]
trimmed_z1 = z1[t_cutoff_i:]
trimmed_z2 = z2[t_cutoff_i:]

N = len(trimmed_t)
print(f"N = {N}")


z1amp = np.fft.rfft(trimmed_z1)
z2amp = np.fft.rfft(trimmed_z2)
freq = np.fft.rfftfreq(len(trimmed_t), 1.0/samplerate)



z1absamp = np.abs(z1amp)
z2absamp = np.abs(z2amp)

fig, axs = plt.subplots(1,2)

axs[0].plot(freq, z1absamp, ".-", label="$z_1$ abs amplitud")
axs[0].plot(freq, z2absamp, ".-", label="$z_2$ abs amplitud")
axs[0].plot(np.array([min(freq), max(freq)]), np.array([0.0, 0.0]), "--k")
axs[0].set_xlabel("frekvens [Hz]")
axs[0].set_ylabel("amplitud [m]")
axs[0].legend()

axs[1].plot(t, z1)
axs[1].plot(t, z2)
axs[1].set_xlabel("tid [s]")
axs[1].set_ylabel("position [m]")

plt.show()
