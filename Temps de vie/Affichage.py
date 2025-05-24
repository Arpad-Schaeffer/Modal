import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import scipy.stats as st
from matplotlib.ticker import MultipleLocator, AutoLocator, AutoMinorLocator

fig, ax = plt.subplots(figsize=(10,6))

# -----------------------------
# Lecture des données
# -----------------------------
filename = "Temps de vie/Data/etalongaev2.csv"  # Remplacez par le chemin réel
skip_lines = 0  # Nombre de lignes à ignorer au début du fichier
try:
    data = pd.read_csv(filename, skiprows=skip_lines)
except Exception as e:
    print(f"Erreur lors de la lecture de {filename}: {e}")
    exit(1)

deltaT = data['deltaT']
channel = data['channel']
err_channel = data['error']

err_deltaT = 0.001

# grille fine pour la courbe
x_fit = np.linspace(deltaT.min(), deltaT.max(), 500)


# modélisation linéaire de channel en fonction de deltaT
def linear(x, a, b):
    return a * x + b

# estimation initiale des paramètres (pente, intercept)
p0 = [
    (channel.max() - channel.min()) / (deltaT.max() - deltaT.min()),
    channel.mean()
]

params, cov = curve_fit(
    linear,
    deltaT, channel,
    sigma=err_channel,
    absolute_sigma=True,
    p0=p0
)

# calcul des incertitudes et des p‑values
se = np.sqrt(np.diag(cov))
# degrés de liberté
dof = len(deltaT) - len(params)

# t-values et p-values
t_vals = params / se
p_vals = 2 * (1 - st.t.cdf(np.abs(t_vals), dof))

# χ² et χ²/ndf
y_model = linear(deltaT, *params)
chi2 = np.sum(((channel - y_model) / err_channel)**2)
red_chi2 = chi2 / dof

# R²
ss_res = np.sum((channel - y_model)**2)
ss_tot = np.sum((channel - np.mean(channel))**2)
r2 = 1 - ss_res / ss_tot

# courbe de fit
y_fit = linear(x_fit, *params)
ax.plot(
    x_fit, y_fit, '-',  color='k'
    # label supprimé pour usage dans textbox
)

# -----------------------------
# Construction du figure 
# -----------------------------
ax.errorbar(
    deltaT, channel, yerr=err_channel, xerr=err_deltaT,
    fmt='o',color='C1', ms=5, capsize=3, label='Calibration'
)
    

# légende des points en haut à gauche sans cadre, avec marge
ax.legend(
    loc='upper left',
    fontsize=10,
    borderpad=0.5,           # espace entre le texte et le bord de la légende
    bbox_to_anchor=(0.03, 0.95)  # décalage depuis le coin supérieur gauche
)

# encadré texte en haut à droite, à l’intérieur du graphe
textstr = (
    "Ajustement linéaire\n"
    f"channel = {params[0]:.3f}·ΔT + {params[1]:.3f}\n"
    f"Pente (A) = {params[0]:.3f} ± {se[0]:.3f} (p={p_vals[0]:.2f})\n"
    f"Ord. à l'origine (B) = {params[1]:.3f} ± {se[1]:.3f} (p={p_vals[1]:.2f})\n"
    f"χ²/ndf = {red_chi2:.2f}\n"
    f"R² = {r2:.2f}"
)

ax.text(
    0.95, 0.05, textstr,
    transform=ax.transAxes, fontsize=10,
    verticalalignment='bottom', horizontalalignment='right',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)


ax.set_ylabel("Channel (proportionnel au délai)")
ax.set_xlabel("deltaT (us)")
ax.set_title("Calibration du canal en fonction du délai")
ax.legend(fontsize=11, loc='upper left', frameon=False, bbox_to_anchor=(0.02, 0.98), borderaxespad=1)
ax.grid(False)


# Graduations automatiques sur les deux axes
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

plt.tight_layout()
plt.show()