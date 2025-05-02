import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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

# Liste des couples (tension, dect) à analyser
couples = [
    (1900, 1),
    (1950, 1),
    (2000, 1)
    # Ajoutez ici d'autres couples spécifiques si nécessaire
]

# Fonction exponentielle pour le fit : f(x) = a * exp(-b * x) + c
def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

# ---------------------------------------------------------------------------
# Section 2 : Configuration de la figure
# ---------------------------------------------------------------------------
plt.figure(figsize=(12, 7))
n_curves = len(couples)
cmap = plt.get_cmap('jet', n_curves)
curve_idx = 0

# ---------------------------------------------------------------------------
# Section 3 : Itération sur les jeux de données pour chaque couple tension-dect
# ---------------------------------------------------------------------------
for V, d in couples:
    filename = f"Threshold/Data/{V}_DECT{d}.csv"
    try:
        data_dect = pd.read_csv(filename)
    except Exception as e:
        print(f"Erreur lors de la lecture de {filename}: {e}")
        continue

    # Lecture des données
    x = data_dect['Tresh (V)']
    y = data_dect['N']

    # Définition des erreurs : erreur sur N (sqrt(N)) et erreur fixe sur Treshold
    y_errors = np.sqrt(y)
    x_errors = np.full_like(x, 0.005)

    # Ajustement des données par régression exponentielle
    initial_guesses = [12000, 10, 300]
    try:
        params, covariance = curve_fit(exp_func, x, y, p0=initial_guesses, maxfev=5000)
    except Exception as e:
        print(f"Erreur dans le fit pour {filename}: {e}")
        continue

    a, b, c = params

    # Génération de la courbe ajustée pour l'affichage
    x_fit = np.linspace(xmin_fit, xmax_fit, 500)
    y_fit = exp_func(x_fit, a, b, c)
    
    # -----------------------------------------------------------------------
    # Calcul du chi² : χ² = Σ[((observé - ajusté)/erreur)²]
    # -----------------------------------------------------------------------
    chi2 = np.sum(((y - exp_func(x, a, b, c)) / y_errors) ** 2)

    # Couleur pour la courbe
    color = cmap(curve_idx)
    curve_idx += 1

    # -----------------------------------------------------------------------
    # Tracé des données et de l'ajustement :
    # -----------------------------------------------------------------------
    plt.errorbar(x, y, yerr=y_errors, xerr=x_errors, fmt='o', color=color, 
                 ecolor='black', capsize=3, markersize=3, alpha=0.7)
    plt.plot(x_fit, y_fit, '-', color=color, lw=2,
             label=f'V={V}, DECT={d} (a={a:.1f}, b={b:.2g}, c={c:.1f}, χ²={chi2:.1f})')
    if log:
        plt.yscale('log')


# ---------------------------------------------------------------------------
# Section 4 : Configuration finale et affichage
# ---------------------------------------------------------------------------
plt.xlabel("Treshold (V)")
plt.ylabel("Counts N")
plt.title("Régression exponentielle pour différents couples tension-dect")
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.legend(fontsize=9)
plt.grid(True)
plt.show()