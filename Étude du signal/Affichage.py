import matplotlib.pyplot as plt
import pandas as pd

# Exemple : chargement de données CSV exportées depuis un oscilloscope
data = pd.read_csv("Étude du signal/Data/Data_04-04-2025_14-54_CH1_CH3.csv")
time = data['Time (ns)']
signal_ch1 = data['CH1']
signal_ch2 = data['CH2']

plt.figure(figsize=(10,5))
plt.plot(time, signal_ch1, label='Canal 1 (scintillateur 1)')
plt.plot(time, signal_ch2, label='Canal 2 (scintillateur 2)', alpha=0.7)
plt.xlabel("Temps (ns)")
plt.ylabel("Amplitude (V)")
plt.title("Signaux typiques détectés sur l’oscilloscope")
plt.legend()
plt.grid(True)
plt.figtext(0.5, -0.1, "Figure: Signaux typiques détectés sur l’oscilloscope", wrap=True, horizontalalignment='center', fontsize=10)  # Légende de la figure
plt.show()