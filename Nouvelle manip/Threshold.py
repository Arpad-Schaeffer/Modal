import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoLocator, AutoMinorLocator


# Option : mettre à True pour échelle log en ordonnée

# Option : définir limites axe x (None pour auto)
xmin = 0.1   # ex: 0.0
xmax = None   # ex: 5.0

# Chemin vers votre fichier CSV (à ajuster)
csv_file = "Nouvelle manip/Threshold_dect1.csv"

# Lecture du CSV en sautant les deux premières lignes
df = pd.read_csv(csv_file, skiprows=2)

# Extraction des données
x = df['Threshold']
y = df['Count']



# application de l'échelle log si demandée




fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x, y, '-o', label="Données brutes", markersize=5)


textstr = (
    r"$\bf{TREX}$"
    f"\n"
        
    )
ax.text(
        0.95, 0.95, textstr,
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    # Configuration du graphique
ax.set_xlabel("Seuil")
ax.set_ylabel("Nombre de coups (N)")
ax.set_title("Reponse des scintillateurs en fonction du seuil")
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


