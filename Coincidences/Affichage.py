import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# -----------------------------
# Lecture des données
# -----------------------------
filename = "Coincidences/Data/DECT1_DECT2_DECT3_Delay.csv"  # Remplacez par le chemin réel
skip_lines = 21  # Nombre de lignes à ignorer au début du fichier
try:
    data = pd.read_csv(filename, skiprows=skip_lines)
except Exception as e:
    print(f"Erreur lors de la lecture de {filename}: {e}")
    exit(1)

# Paramètres
num_scint = 3       # 2 ou 3 scintillateurs
log_scale = False   # True pour échelle log y
theorical = False    # True pour afficher la courbe théorique

# Regroupement et moyennes
grouped = data.groupby('Width', as_index=False).mean()
width_ns      = grouped['Width']              # en ns
width_s       = width_ns * 1e-9               # si nécessaire en s
count_WO      = grouped['Count_WO']           # sur 10 min
count_random = grouped['Count']               # sur 10 min

# Passage en compte/minute
count_WO_min       = count_WO      / 10
count_random_min   = count_random  / 10
count_diff_min     = (count_WO - count_random) / 10
ratio = count_diff_min / count_random_min

# Erreurs Poisson
err_WO     = np.sqrt(count_WO)     / 10
err_random= np.sqrt(count_random)  / 10
err_diff   = np.sqrt(err_WO**2 + err_random**2) / 10
err_ratio = np.sqrt((err_diff/count_diff_min)**2 + (err_random/ratio)**2)

# Théorie
R1_Hz, R2_Hz, R3_Hz = 0.5, 0.5, 0.5
T = 10  # minutes => si vos R sont en Hz, préférez T=600
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
    y_fit = (K * x_fit) / 10
else:
    y_fit = (K * x_fit**2) / 10

# -----------------------------
# Construction du figure à deux panneaux
# -----------------------------
fig, (ax1, ax2) = plt.subplots(
    2, 1,
    sharex=True,
    gridspec_kw={'height_ratios': [3, 1]},
    figsize=(10, 7)
)

# --- Panel du haut : données + théorie ---
ax1.errorbar(
    width_ns, count_WO_min, yerr=err_WO,
    fmt='o', ms=5, capsize=3, label='Coïncidences totales'
)
ax1.errorbar(
    width_ns, count_random_min, yerr=err_random,
    fmt='s', ms=5, capsize=3, label='Coïncidences aléatoires'
)
ax1.errorbar(
    width_ns, count_diff_min, yerr=err_diff,
    fmt='s', ms=5, capsize=3, label='Différence (WO - random)'
)
if theorical:
    ax1.plot(
        x_fit, y_fit,
        '-', lw=2, color='C3',
        label=f'Courbe théorique ({num_scint} scint.)'
    )

ax1.set_ylabel("Nombre de coïncidences par minute")
if log_scale:
    ax1.set_yscale('log')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, which='both', ls='--', alpha=0.3)

# --- Panel du bas : résidus ---
if theorical:
    # interpolation pour évaluer y_fit aux abscisses width_ns
    interp_th = interp1d(x_fit, y_fit, bounds_error=False, fill_value=0)
    resid = count_WO_min - interp_th(width_ns)
    err_resid = err_WO
else:
    resid = count_diff_min
    err_resid = np.sqrt(err_WO**2 + err_random**2)

ax2.errorbar(
    width_ns, ratio, yerr=err_ratio,
    fmt='o', ms=5, capsize=3, color='k'
)
ax2.axhline(0, color='k', lw=1)
ax2.set_xlabel("Largeur de la fenêtre de coïncidence (ns)")
ax2.set_ylabel("Ratio")
ax2.grid(True, which='both', ls='--', alpha=0.3)

# --- Réglages finaux ---
xmin, xmax = width_ns.min()*0.8, width_ns.max()*1.2
ymin, ymax = None, None  # ou définir manuellement
ax1.set_xlim(xmin, xmax)
ax2.set_xlim(xmin, xmax)
if ymin is not None and ymax is not None:
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)

plt.tight_layout(h_pad=0.1)
plt.show()