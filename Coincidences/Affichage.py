import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os
from datetime import datetime
from matplotlib.ticker import AutoLocator, AutoMinorLocator

save_fig = True

# -----------------------------
# Lecture des données
# -----------------------------
filename = "Coincidences/Data/DECT1_DECT2_Delay.csv"  # Remplacez par le chemin réel
skip_lines = 29  # Nombre de lignes à ignorer au début du fichier
try:
    data = pd.read_csv(filename, skiprows=skip_lines)
except Exception as e:
    print(f"Erreur lors de la lecture de {filename}: {e}")
    exit(1)

# Paramètres
num_scint = 2       # 2 ou 3 scintillateurs
log_scale = False   # True pour échelle log y
theorical = False    # True pour afficher la courbe théorique

# Regroupement et moyennes
grouped = data.groupby('Width', as_index=False).mean()
width_ns      = grouped['Width']              # en ns
width_s       = width_ns * 1e-9               # si nécessaire en s
count_WO      = grouped['Count_WO']           # sur 4 min
count_random = grouped['Count']               # sur 4 min

# Passage en compte/minute
count_WO_min       = count_WO      / 4
count_random_min   = count_random  / 4
count_diff_min     = (count_WO - count_random) / 4
ratio = count_diff_min / count_random_min

# Erreurs Poisson
err_WO     = np.sqrt(count_WO)     / 4
err_random= np.sqrt(count_random)  / 4
err_diff   = np.sqrt(err_WO**2 + err_random**2)
err_ratio = np.sqrt((err_diff/count_diff_min)**2 + (err_random/ratio)**2)

# Théorie
R1_Hz, R2_Hz, R3_Hz = 0.5, 0.5, 0.5
T = 4  # minutes => si vos R sont en Hz, préférez T=600
# coefficient
if num_scint == 2:
    K = 2 * R1_Hz * R2_Hz * T
elif num_scint == 3:
    K = 4 * R1_Hz * R2_Hz * R3_Hz * T
else:
    raise ValueError("num_scint doit être 2 ou 3")
# grille fine pour la courbe
x_fit = np.linspace(0, width_ns.max(), 500)
if num_scint == 2:
    y_fit = (K * x_fit) / 4
else:
    y_fit = (K * x_fit**2) / 4

# -----------------------------
# Construction du figure à deux panneaux (style publication)
# -----------------------------
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    sharex=True,
    gridspec_kw={'height_ratios': [3, 1]},
    figsize=(10, 7)
)

# --- Panel du haut : données (style publication) ---
ax1.errorbar(
    width_ns, count_WO_min, yerr=err_WO, fmt='o', label="Coïncidences totales", markersize=5, color='k', alpha=0.7, capsize=3
)
ax1.errorbar(
    width_ns, count_random_min, yerr=err_random, fmt='s', label="Coïncidences aléatoires", markersize=5, color='C0', alpha=0.7, capsize=3
)
ax1.errorbar(
    width_ns, count_diff_min, yerr=err_diff, fmt='D', label="Différence (totales - aléatoires)", markersize=5, color='C1', alpha=0.7, capsize=3
)
if theorical:
    ax1.plot(
        x_fit, y_fit,
        '-', lw=2, color='C3',
        label=f'Courbe théorique ({num_scint} scint.)'
    )

# Encadré annotation (exemple, à adapter)
textstr = (
    r"$\bf{TREX}$"
    f"\n"
    f"Nombre de scintillateurs : {num_scint}"
    f"\n"
    f"Ratio = (coïncidences totales - aléatoires) / aléatoires"
    f"\n"
    f"= nombre de vraies coïncidences / bruit de fond)"
)
ax1.text(
    0.97, 0.95, textstr,
    transform=ax1.transAxes, fontsize=10,
    verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)

ax1.set_xlabel("")
ax1.set_ylabel("Coïncidences par minute", fontsize=13)
ax1.set_title("Étude des coïncidences en fonction de la largeur de fenêtre", fontsize=14)
ax1.legend(fontsize=11, loc='upper left', frameon=False, bbox_to_anchor=(0.02, 0.98), borderaxespad=1)
ax1.grid(False)

# Axes publication-ready
ax1.xaxis.set_major_locator(AutoLocator())
ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_major_locator(AutoLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.tick_params(axis='both', which='major', direction='in', top=True, right=True, length=10)
ax1.tick_params(axis='both', which='minor', direction='in', top=True, right=True, length=5)

# Ajustement de la limite supérieure de l'axe y
current_ymin, current_ymax = ax1.get_ylim()
new_ymax = current_ymax + (current_ymax - current_ymin) * 0.4
ax1.set_ylim(current_ymin, new_ymax)

# --- Panel du bas : ratio (publication-ready) ---
ax2.errorbar(
    width_ns, ratio, yerr=err_ratio,
    fmt='o', ms=5, capsize=3, color='k', alpha=0.7
)
ax2.axhline(0, color='k', lw=1)
ax2.set_xlabel("Largeur de la fenêtre de coïncidence (ns)", fontsize=13)
ax2.set_ylabel("Ratio", fontsize=13)
ax2.grid(False)
ax2.xaxis.set_major_locator(AutoLocator())
ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_major_locator(AutoLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax2.tick_params(axis='both', which='major', direction='in', top=True, right=True, length=10)
ax2.tick_params(axis='both', which='minor', direction='in', top=True, right=True, length=5)

# Limites x
xmin, xmax = width_ns.min()*0.8, width_ns.max()*1.2
ax1.set_xlim(xmin, xmax)
ax2.set_xlim(xmin, xmax)

plt.tight_layout(h_pad=0.1)

# Sauvegarde automatique dans Latex/Images

def make_filename_from_csv(csv_path):
    base = os.path.basename(csv_path)
    name, _ = os.path.splitext(base)
    return f"Latex/Images/Coincidences_2_Scintillateurs.png"

if save_fig:
    output_path = make_filename_from_csv(filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé dans : {output_path}")

plt.show()