import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator, AutoMinorLocator

# Paramètres utilisateur
filename = "Temps de vie/Data/Data_09-05-2025_12-23_CH2_CH3_CH4.csv"
save_fig = True
num_scint = 3  # À adapter si besoin

# Lecture des données
# On suppose que chaque ligne correspond à une mesure, on va simuler des largeurs de fenêtre et des ratios pour l'exemple

data = pd.read_csv(filename)

# Affichage d'un seul graphe : tension (V) en fonction du temps (s) pour chaque canal
fig, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(data['Temps (s)'], data['CH2 (V)'], label='CH2', color='tab:blue')
ax1.plot(data['Temps (s)'], data['CH3 (V)'], label='CH3', color='tab:orange')
ax1.plot(data['Temps (s)'], data['CH4 (V)'], label='CH4', color='tab:green')

ax1.set_xlabel('Temps (s)', fontsize=13)
ax1.set_ylabel('Tension (V)', fontsize=13)
ax1.set_title('Tension en fonction du temps pour chaque canal', fontsize=14)

# Nouveau texte d'annotation et titre
textstr = (
    r"$\bf{TREX}$"
    f"\n"
    f"Expérience TAC (Time-to-Amplitude Converter)"
)
ax1.set_title("Signaux de coïncidence et sortie TAC : mesure du temps de vie du muon", fontsize=14)
ax1.text(
    0.97, 0.95, textstr,
    transform=ax1.transAxes, fontsize=10,
    verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)

ax1.legend(fontsize=11, loc='upper left', frameon=False, bbox_to_anchor=(0.02, 0.98), borderaxespad=1)

# Axes publication-ready
ax1.xaxis.set_major_locator(AutoLocator())
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_major_locator(AutoLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.tick_params(axis='both', which='major', direction='in', top=True, right=True, length=10)
ax1.tick_params(axis='both', which='minor', direction='in', top=True, right=True, length=5)

# Désactivation de la grille
ax1.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.7)
ax1.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)

# Ajustement de la limite supérieure de l'axe y
current_ymin, current_ymax = ax1.get_ylim()
new_ymax = current_ymax + (current_ymax - current_ymin) * 0.4
ax1.set_ylim(current_ymin, new_ymax)

plt.tight_layout()

if save_fig:
    output_path = "Latex/Images/TAC.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé dans : {output_path}")

plt.show()
