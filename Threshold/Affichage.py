import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from datetime import datetime
from matplotlib.ticker import AutoLocator, AutoMinorLocator

# ---------------------------------------------------------------------------
# Section 1 : Paramètres et définition de la fonction de régression
# ---------------------------------------------------------------------------
# Ajoutez ici vos bornes pour les axes
xmin = 0
xmax = 1
ymin = 0
ymax = 30000
xmin_fit = 0.01
xmax_fit = 1
log = False # Si True, on utilise une échelle logarithmique pour l'axe des ordonnées
save_fig = True

# Liste des couples (tension, dect) à analyser
couples = [
    (1900, 1),
    (1950, 1),
    (2000, 1)
    # Ajoutez ici d'autres couples spécifiques si nécessaire
]

# Fonction exponentielle pour le fit : f(x) = a * exp(-x / tau) + c
# Remplace b par tau dans la fonction de fit et l'affichage

def exp_func(x, a, tau, c):
    return a * np.exp(-x / tau) + c

# ---------------------------------------------------------------------------
# Section 2 : Tracé publication-ready avec toutes les tensions sur la même courbe
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
colors = plt.get_cmap('jet', len(couples))

legend_handles = []
fit_params_text = []

for idx, (V, d) in enumerate(couples):
    filename = f"Threshold/Data/{V}_DECT{d}.csv"
    try:
        data_dect = pd.read_csv(filename)
    except Exception as e:
        print(f"Erreur lors de la lecture de {filename}: {e}")
        continue

    x = data_dect['Tresh (V)']
    y = data_dect['N']
    y_errors = np.sqrt(y)
    x_errors = np.full_like(x, 0.01)  # Erreur sur le seuil : ±0.01 V

    initial_guesses = [12000, 0.1, 300]  # tau initialisé à 0.1 (ajustez si besoin)
    try:
        params, covariance = curve_fit(exp_func, x, y, p0=initial_guesses, maxfev=5000)
    except Exception as e:
        print(f"Erreur dans le fit pour {filename}: {e}")
        continue

    a, tau, c = params
    x_fit = np.linspace(xmin_fit, xmax_fit, 500)
    y_fit = exp_func(x_fit, a, tau, c)
    chi2 = np.sum(((y - exp_func(x, a, tau, c)) / y_errors) ** 2)

    color = colors(idx)
    # Tracé des points expérimentaux avec barres d'erreur Poisson (sqrt(N))
    h_data = ax.errorbar(x, y, yerr=y_errors, xerr=x_errors, fmt='o', label=f"Données V={V}V", markersize=4, color=color, alpha=0.7, capsize=3, ecolor=color)
    # Tracé du fit
    h_fit, = ax.plot(x_fit, y_fit, '-', label=f"Régression V={V}V", color=color, linewidth=2)
    legend_handles.extend([h_data, h_fit])
    fit_params_text.append(f"V={V}V : a={a:.2g}, $\\tau$={tau:.2g}, c={c:.2g}, χ²={chi2:.2f}")

# Encadré annotation général (en haut à droite)
textstr = (
    r"$\bf{TREX}$"
    f"\n"
    r"$y = a e^{-x/\tau} + c$" + "\n"
    + "\n".join(fit_params_text)
)
ax.text(
    0.97, 0.95, textstr,
    transform=ax.transAxes, fontsize=10,
    verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
)

ax.set_xlabel("Seuil (V)", fontsize=13)
ax.set_ylabel("Nombre de counts (N)", fontsize=13)
ax.set_title("Fit exponentiel sur N(Seuil) pour différentes tensions", fontsize=14)
ax.legend(handles=legend_handles, fontsize=10, loc='upper left', frameon=False, bbox_to_anchor=(0.02, 0.98))
ax.grid(False)

ax.xaxis.set_major_locator(AutoLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_major_locator(AutoLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', which='major', direction='in', top=True, right=True, length=10)
ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True, length=5)
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.relim()
ax.autoscale_view()
current_xmin, current_xmax = ax.get_xlim()
current_ymin, current_ymax = ax.get_ylim()
new_xmin = current_xmin - (current_xmax - current_xmin) * 0.35
new_ymax = current_ymax + (current_ymax - current_ymin) * 0.4
ax.set_xlim(new_xmin, current_xmax)
ax.set_ylim(current_ymin, new_ymax)

if save_fig:
    output_path = "Latex/Images/Threshold_Scintillateur_1.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé dans : {output_path}")

plt.tight_layout()
plt.show()
