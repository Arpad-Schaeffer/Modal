import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.ticker import MultipleLocator, AutoLocator, AutoMinorLocator


def lire_spectre_spe(filepath):
    with open(filepath, "r") as f:
        lignes = f.readlines()

    # Trouver l'index où commencent les données (après $DATA:)
    index_data = None
    data_range = None
    for i, ligne in enumerate(lignes):
        if ligne.strip().startswith("$DATA:"):
            index_data = i + 1  # données à partir de la ligne suivante
            # Lire la plage de données (ex: "0 2047")
            try:
                data_range = list(map(int, lignes[index_data].strip().split()))
                if len(data_range) != 2:
                    raise ValueError("Format incorrect pour la plage de données après $DATA:")
            except Exception as e:
                raise ValueError(f"Erreur lors de la lecture de la plage de données: {e}")
            index_data += 1  # Passer à la ligne suivante après la plage
            break

    if index_data is None:
        raise ValueError("Pas de section $DATA: trouvée")

    # Extraire les valeurs de comptage
    compteurs = []
    for ligne in lignes[index_data:]:
        ligne = ligne.strip()
        if ligne == "" or ligne.startswith("$"):  # fin des données
            break
        try:
            compteurs.append(int(ligne))
        except ValueError:
            raise ValueError(f"Valeur non valide trouvée dans les données: {ligne}")

    # Vérifier que le nombre de données correspond à la plage spécifiée
    if len(compteurs) != (data_range[1] - data_range[0] + 1):
        raise ValueError(
            f"Le nombre de données ({len(compteurs)}) ne correspond pas à la plage spécifiée ({data_range})"
        )

    return compteurs

# Fonction exponentielle pour le fit
def fonction_exponentielle(x, a, b, c):
    return a * np.exp(-b * x) + c

# Exemple d'utilisation
fichier = "Temps de vie/3dayslifetimespe.Spe"
try:
    donnees = lire_spectre_spe(fichier)

    # Limiter les données aux canaux 0 à 100
    x_data = np.arange(20, 101)
    y_data = donnees[20:101]

    # Ajustement exponentiel
    params, _ = curve_fit(fonction_exponentielle, x_data, y_data, p0=(1, 0.01, 1))

    # Générer les données ajustées
    y_fit = fonction_exponentielle(x_data, *params)

    # Affichage des données et du fit
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x_data, y_data, 'o', label="Données brutes", markersize=5)
    ax.plot(x_data, y_fit, '-', label="Fit exponentiel", color="red")

    #calcul de tau
    tau = (1 / params[1]-16.8)/203

    textstr = (
        r"$\bf{TREX}$"
        f"\n"
        r" Fit exponentiel : $N = ae^{-bx}+c$"
        f"\n"
        f"a = {params[0]:.3f}\n"
        f"b = {params[1]:.3f}\n"
        f"c = {params[2]:.3f}\n"
        f"tau = {tau:.3f}"
    )
    ax.text(
        0.95, 0.95, textstr,
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    # Configuration du graphique
    ax.set_xlabel("Canal (x)")
    ax.set_ylabel("Nombre de coups (N)")
    ax.set_title("Spectre extrait de .spe (Maestro), Aquisition sur trois jours")
    ax.legend()
    ax.grid(False)
    # graduations automatiques sur les deux axes
    ax.xaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    # ticks sur tous les côtés, vers l'intérieur
    ax.tick_params(axis='both', which='major', direction='in', top=True, right=True, length=10)
    ax.tick_params(axis='both', which='minor', direction='in', top=True, right=True, length=5)

    # échelle automatique
    ax.relim()
    ax.autoscale_view()

    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Erreur : {e}")
