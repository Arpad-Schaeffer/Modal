import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator, AutoMinorLocator
import numpy as np

# Charger les données à partir de la ligne 12
file_path = "Energie\energie.csv"
data = pd.read_csv(file_path, skiprows=11)

# Vérifier les colonnes disponibles
print("Colonnes disponibles :", data.columns)

# Supposons que les colonnes sont nommées "Abscisse" et "Valeur"
data.columns = ["Epaisseur", "N"]


# Moyenne des valeurs pour une même abscisse
data_grouped = data.groupby("Epaisseur", as_index=False).agg(
    N_mean=("N", "mean"),
    N_std=("N", "std"),
    count=("N", "count")
)
# Calculer l'incertitude de Poisson avant de diviser par 5
data_grouped["Poisson_uncertainty"] = np.sqrt(data_grouped["N_mean"])

# Diviser N par 5 pour obtenir le nombre de coups par minute
data_grouped["N_mean"] = data_grouped["N_mean"] / 5
data_grouped["N_std"] = data_grouped["N_std"] / 5
data_grouped["Poisson_uncertainty"] = data_grouped["Poisson_uncertainty"] / 5


# Calcul des incertitudes max
data_grouped["Uncertainty"] = data_grouped[["Poisson_uncertainty", "N_std"]].max(axis=1)

# Afficher les incertitudes pour diagnostic
print(data_grouped[["Epaisseur", "N_mean", "Poisson_uncertainty", "N_std", "Uncertainty"]])

# Ajouter une incertitude fixe de 0.5 sur l'axe des x
x_uncertainty = 0.5

# Tracer les données avec les barres d'erreur (y et x)
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(
    data_grouped["Epaisseur"],
    data_grouped["N_mean"],
    xerr=x_uncertainty,  # Incertitude sur l'axe des x
    yerr=data_grouped["Uncertainty"],  # Incertitude sur l'axe des y
    fmt='o',
    linestyle='--',
    color='C1',
    capsize=5,
    label="Données moyennes avec incertitudes"
)

# Ajouter le texte du fit
textstr_trimmed = (
    r"$\bf{TREX}$"
    f"\n"
    f"Evolution du nombre de coup par minutes en fonction de l'épaisseur\n"
)
ax.text(
    0.95, 0.95, textstr_trimmed,
    transform=ax.transAxes, fontsize=10,
    verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

# Configuration du graphique
ax.set_xlabel("Epaisseur (mm)")
ax.set_ylabel("Nombre de coups par minute (N)")
ax.set_title("Evolution du nombre de coup par minutes en fonction de l'épaisseur")
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