import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator, AutoMinorLocator
import numpy as np

# Charger les données
file_path = "Energie/energie.csv"
data = pd.read_csv(file_path, skiprows=11)

# Nommer les colonnes si nécessaire
data.columns = ["Epaisseur", "N"]

# Moyenne des valeurs pour une même épaisseur
data_grouped = data.groupby("Epaisseur", as_index=False).agg(
    N_mean=("N", "mean"),
    N_std=("N", "std"),
    count=("N", "count")
)

# Calcul des incertitudes (Poisson + std)
data_grouped["Poisson_uncertainty"] = np.sqrt(data_grouped["N_mean"])
data_grouped["N_mean"] /= 5
data_grouped["N_std"] /= 5
data_grouped["Poisson_uncertainty"] /= 5
data_grouped["Uncertainty"] = data_grouped[["Poisson_uncertainty", "N_std"]].max(axis=1)

# --- Modélisation pour la figure 3 ---

# Estimation de l'énergie minimale en MeV (x en mm → cm)
data_grouped["E_min"] = 19.3 * data_grouped["Epaisseur"] / 10

# Filtres : retirer points à N = 0 ou E_min = 0 (log impossible)
filtered = data_grouped[(data_grouped["N_mean"] > 0) & (data_grouped["E_min"] > 0)].copy()

# Ignorer le premier et le deuxième point
filtered = filtered.iloc[list(range(2, len(filtered)))].copy()

# Passer au log
filtered["ln_Emin"] = np.log(filtered["E_min"])
filtered["ln_N"] = np.log(filtered["N_mean"])

# Propagation des incertitudes avec le logarithme
filtered["ln_Emin_uncertainty"] = filtered["Uncertainty"] / filtered["E_min"]
filtered["ln_N_uncertainty"] = filtered["Uncertainty"] / filtered["N_mean"]

# Ajustement linéaire pondéré
weights = 1 / filtered["ln_N_uncertainty"]  # Poids inverses des incertitudes sur ln(N)
coeffs, cov_matrix = np.polyfit(filtered["ln_Emin"], filtered["ln_N"], deg=1, w=weights, cov=True)

# Extraction des coefficients et des incertitudes
slope, intercept = coeffs
slope_uncertainty = np.sqrt(cov_matrix[0, 0])  # Incertitude sur la pente
gamma = 1 - slope  # Calcul de gamma
gamma_uncertainty = slope_uncertainty  # Incertitude sur gamma (identique à celle de la pente)

# Mise à jour de l'étiquette avec les incertitudes
fit_label = (
    rf"$\ln N = {slope:.2f} \ln E_{{\min}} + {intercept:.2f}$"
    f"\n"
    rf"$\gamma = {gamma:.2f} \pm {gamma_uncertainty:.2f}$"
)

# Droite ajustée
x_fit = np.linspace(filtered["ln_Emin"].min(), filtered["ln_Emin"].max(), 100)
y_fit = slope * x_fit + intercept

# Tracé
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(
    filtered["ln_Emin"],
    filtered["ln_N"],
    yerr=filtered["Uncertainty"] / filtered["N_mean"],  # erreur relative propagée
    fmt='o', color='C1', capsize=5,
    label="Données expérimentales"
)
ax.plot(x_fit, y_fit, 'k--', label=fit_label)

textstr_trimmed = (
    r"$\bf{TREX}$"
    f"\n"
    r"Régression linéaire de $\ln N$ en fonction de $\ln E_{\min}$"
    f"\n"
    r"avec $\gamma = 1 - \text{slope}$"
    f"\n"
    rf"$\gamma = {gamma:.2f} \pm {gamma_uncertainty:.2f}$"
)
ax.text(
    0.95, 0.95, textstr_trimmed,
    transform=ax.transAxes, fontsize=10,
    verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

# Légende et axes
ax.set_xlabel(r"$\ln(E_{\min})$ (MeV)")
ax.set_ylabel(r"$\ln(N)$")
ax.set_title(r"Régression linéaire de $\ln N$ en fonction de $\ln E_{\min}$")
ax.legend()
ax.grid(True)

# Mise en forme
ax.xaxis.set_major_locator(AutoLocator())
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_major_locator(AutoLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(axis='both', which='major', direction='in', top=True, right=True)
ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True)

plt.tight_layout()
plt.show()
