import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Chemin vers le fichier CSV avec header
csv_file = "Étude du signal/Data/Data_04-04-2025_14-54_CH1_CH3.csv"  # Remplacez par le chemin réel

# Lecture du CSV avec header
df = pd.read_csv(csv_file)

# Affichage des premières lignes pour vérifier le contenu
print(df.head())

# Extraction des données pour le graphique
# Ici, "Temps (s)" est utilisé pour l'axe des abscisses, et "CH2 (V)" pour les valeurs
x_data = df["Temps (s)"].values
y_data = df["CH3 (V)"].values

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
regression_mode = "desc"  # Modifiez cette variable pour choisir le mode souhaité

# Définition de la fonction exponentielle à ajuster : y = a * exp(b*x) + c
def expo_func(x, a, b, c):
    return a * np.exp(b * x) + c

# Pour l'ajustement, on cherche des indices correspondant à certaines valeurs dans y_data.
min_index = np.where(np.isclose(y_data, -75, atol=1e-1))[0][0]
max_index = np.where(np.isclose(y_data, -75, atol=1e-2))[0][-1]
print("Indices utilisés :", x_data[min_index], y_data[min_index], x_data[max_index], y_data[max_index])

# Séparation des données en deux parties :
# Descente : du début jusqu'au point de min_index
x_desc = x_data[:min_index+1]
y_desc = y_data[:min_index+1]

# Montée : du point de max_index jusqu'à la fin
x_asc = x_data[max_index:]
y_asc = y_data[max_index:]

# Initialisation des variables
popt_desc = None
popt_asc = None
y_fit_desc = None
y_fit_asc = None
tau_desc = None
tau_asc = None

# Régression sur la descente si demandée
if regression_mode in ["desc", "both"]:
    popt_desc, _ = curve_fit(expo_func, x_desc, y_desc, p0=(-1, 10**9, 0))
    y_fit_desc = expo_func(x_desc, *popt_desc)
    tau_desc = 1.0 / popt_desc[1]
    print("Paramètres pour la descente (a, b, c) :", popt_desc)
    print("Temps caractéristique descente (tau):", tau_desc)

# Régression sur la montée si demandée
if regression_mode in ["asc", "both"]:
    popt_asc, _ = curve_fit(expo_func, x_asc, y_asc, p0=(-1, -10**9, 0))
    y_fit_asc = expo_func(x_asc, *popt_asc)
    tau_asc = 1.0 / popt_asc[1]
    print("Paramètres pour la montée (a, b, c)   :", popt_asc)
    print("Temps caractéristique montée (tau):", tau_asc)

# Tracé des données et des ajustements
plt.figure(figsize=(10, 6))
plt.plot(x_data, y_data, 'ko', markersize=3, label="Données brutes")
if regression_mode in ["desc", "both"]:
    plt.plot(x_desc, y_fit_desc, 'b-', linewidth=2, label="Régression exponentielle (descente)")
if regression_mode in ["asc", "both"]:
    plt.plot(x_asc, y_fit_asc, 'r-', linewidth=2, label="Régression exponentielle (montée)")
plt.xlabel("Temps (s)")
plt.ylabel("CH2 (V)")
plt.title("Graphique des données et régressions exponentielles")
plt.grid(True)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=12, ncol=1)

# Affichage des constantes et de τ sur le graphique, le cas échéant
ax = plt.gca()
if regression_mode in ["desc", "both"]:
    text_desc = (
        "Descente:\n"
        "a=" + f"{popt_desc[0]:.3g}" +
        ", b=" + f"{popt_desc[1]:.3g}" +
        ", c=" + f"{popt_desc[2]:.3g}" + "\n\n" +
        "$\\mathbf{\\tau} = 1/b = " + f"{tau_desc:.3e}" + "$"
    )
    ax.text(0.05, 0.75, text_desc, transform=ax.transAxes, fontsize=10, verticalalignment='top', color='blue')
if regression_mode in ["asc", "both"]:
    text_asc = (
        "Montée:\n"
        "a=" + f"{popt_asc[0]:.3g}" +
        ", b=" + f"{popt_asc[1]:.3g}" +
        ", c=" + f"{popt_asc[2]:.3g}" + "\n\n" +
        "$\\mathbf{\\tau} = 1/b = " + f"{tau_asc:.3e}" + "$"
    )
    ax.text(0.05, 0.60, text_asc, transform=ax.transAxes, fontsize=10, verticalalignment='top', color='red')

plt.tight_layout()
plt.show()