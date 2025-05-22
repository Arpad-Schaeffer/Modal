import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import AutoLocator, AutoMinorLocator
import os
from datetime import datetime



# Chemin vers le fichier CSV avec metadata et données
csv_file = "Data J1/DataJ2Chanel3CSV.CSV"  # Remplacez par le chemin réel

# Lecture du CSV sans header pour inclure metadata et données
df = pd.read_csv(csv_file, header=None)

# Extraction des unités horizontales (temps) et verticales (tension)
unit_time = df.loc[df[0] == "Horizontal Units", 1].iloc[0]
unit_voltage = df.loc[df[0] == "Vertical Units", 1].iloc[0]

# Filtrer les lignes de données (trois premières colonnes vides)
data_df = df[df[0].isna() & df[1].isna() & df[2].isna()].copy()

# Renommer et sélectionner les colonnes de données
data_df = data_df.rename(columns={3: "Temps", 4: "Tension"})
data_df = data_df[["Temps", "Tension"]].astype(float)

# Affichage des premières lignes pour vérifier
print(data_df.head())

# Extraction des données pour le graphique
x_data = data_df["Temps"].values
y_data = data_df["Tension"].values

# Optionnel : tracé brut des données pour vérification
if False:
    plt.figure(figsize=(10, 6))
    plt.plot(x_data, y_data, 'ko', markersize=3, label="Données brutes")
    plt.xlabel("Temps (s)")
    plt.ylabel("CH2 (V)")
    plt.title("Données brutes extraites du CSV")
    plt.grid(True)
    plt.legend()
    plt.show()

# Choix du mode de régression : "desc", "asc" ou "both"
regression_mode = "both"  # Modifiez cette variable pour choisir le mode souhaité

# Définition d'une fonction exponentielle avec tau : y = a * exp(-x/tau) + c

def exp_func(x, a, tau, c):
    return a * np.exp(-x / tau) + c

# Trouver l'indice du minimum global (ymin)
min_index = np.argmin(y_data)
x_min = x_data[min_index]
y_min = y_data[min_index]
print(f"Minimum global : x = {x_min}, y = {y_min}")

# Définir les bornes pour les fits
x_desc_max = x_min - 5e-10
x_asc_min = x_min + 5e-10

# Descente (avant le minimum, jusqu'à x(ymin))
desc_mask = x_data <= x_desc_max
x_desc = x_data[desc_mask]
y_desc = y_data[desc_mask]

# Montée (après le minimum, à partir de x(ymin))
asc_mask = x_data >= x_asc_min
x_asc = x_data[asc_mask]
y_asc = y_data[asc_mask]

# Ajustement exponentielle décroissante sur la descente
popt_desc, _ = curve_fit(exp_func, x_desc, y_desc, p0=(-1, -1e-9, y_desc[-1]))
y_fit_desc = exp_func(x_desc, *popt_desc)
tau_desc = popt_desc[1]
print("Paramètres descente (a, tau, c):", popt_desc)
print("Tau descente:", tau_desc)

# Ajustement exponentielle croissante sur la montée (on inverse le signe de a si besoin)
popt_asc, _ = curve_fit(exp_func, x_asc, y_asc, p0=(-1, 1e-9, y_asc[-1]))
y_fit_asc = exp_func(x_asc, *popt_asc)
tau_asc = popt_asc[1]
print("Paramètres montée (a, tau, c):", popt_asc)
print("Tau montée:", tau_asc)

# Tracé publication style
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_data, y_data, 'o', label="Données brutes", markersize=4, color='k', alpha=0.7)
ax.plot(x_desc, y_fit_desc, '-', label="Fit exp. décroissante", color="blue", linewidth=2)
ax.plot(x_asc, y_fit_asc, '-', label="Fit exp. croissante", color="red", linewidth=2)

# Encadré annotation descente (en haut à gauche)
textstr = (
    r"$\bf{TREX}$"
    f"\n"
    f"Fit : $y = a e^{{-x/\\tau}} + c$"
    f"\n"
    "Descente"
    f"\n"
    f"a = {popt_desc[0]:.3g}, "
    f"$\\tau$ = {tau_desc:.3g} s, "
    f"c = {popt_desc[2]:.3g}"
    f"\n"
    "Montée"
    f"\n"
    f"a = {popt_asc[0]:.3g}, "
    f"$\\tau$ = {tau_asc:.3g} s, "
    f"c = {popt_asc[2]:.3g}"
)
ax.text(
    0.97, 0.95, textstr,
    transform=ax.transAxes, fontsize=10,
    verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)

# Configuration du graphique
ax.set_xlabel(f"Temps ({unit_time})", fontsize=13)
ax.set_ylabel(f"Tension ({unit_voltage})", fontsize=13)
ax.set_title("Fit exponentiel décroissant/croissant sur le signal de détection de muons", fontsize=14)
ax.legend(fontsize=11, loc='upper left', frameon=False, bbox_to_anchor=(0.02, 0.98), borderaxespad=1)
ax.grid(False)

# Graduations automatiques majeures et mineures
ax.xaxis.set_major_locator(AutoLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_major_locator(AutoLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

# Ticks sur tous les côtés, vers l'intérieur
ax.tick_params(axis='both', which='major', direction='in', top=True, right=True, length=10)
ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True, length=5)

# Échelle automatique
ax.relim()
ax.autoscale_view()

# Ajustement de la limite supérieure de l'axe y
current_ymin, current_ymax = ax.get_ylim()
new_ymax = current_ymax + (current_ymax - current_ymin) * 0.4
ax.set_ylim(current_ymin, new_ymax)

# Sauvegarde du graphique en PNG dans Latex/Images avec un nom adapté
def make_filename_from_csv(csv_path):
    base = os.path.basename(csv_path)
    name, _ = os.path.splitext(base)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"Latex/Images/Desexitation_Scintillateur_2.png"

save_fig = True  # Passez à False pour désactiver la sauvegarde
if save_fig:
    output_path = make_filename_from_csv(csv_file)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé dans : {output_path}")

plt.tight_layout()
plt.show()