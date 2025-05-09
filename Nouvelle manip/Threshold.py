import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Option : mettre à True pour échelle log en ordonnée
log = False

# Option : définir limites axe x (None pour auto)
xmin = 0.1   # ex: 0.0
xmax = None   # ex: 5.0

# Chemin vers votre fichier CSV (à ajuster)
csv_file = "/Users/malotamalet/Desktop/2A/Modal/Modal/Nouvelle manip/Threshold.csv"

# Lecture du CSV en sautant les deux premières lignes
df = pd.read_csv(csv_file, skiprows=2)

# Extraction des données
x = df['Threshold']
y = df['Count']

# Tracé
plt.figure(figsize=(8, 5))
plt.plot(x, y, marker='o', linestyle='-')

# application de l'échelle log si demandée
if log:
    plt.yscale('log')

# application des limites sur l'axe x si demandées
if xmin is not None or xmax is not None:
    plt.xlim(xmin, xmax)
    # mise à jour automatique de l'axe y selon la plage x visible
    mask = pd.Series(True, index=x.index)
    if xmin is not None:
        mask &= x >= xmin
    if xmax is not None:
        mask &= x <= xmax
    y_vis = y[mask]
    if not y_vis.empty:
        plt.ylim(y_vis.min(), y_vis.max())

plt.xlabel('Threshold (V)')
plt.ylabel('Count')
plt.title('Count en fonction du Threshold' + (' [log]' if log else ''))
plt.grid(True)
plt.show()