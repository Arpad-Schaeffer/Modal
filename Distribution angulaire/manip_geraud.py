import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.ticker import MultipleLocator, AutoLocator, AutoMinorLocator
 
# Fonction cosinus carré pour le fit
def cos_squared(x, a, b, c):
    return a * (np.cos(np.radians(x) + b) ** 2) + c

# Ouvrir et lire le fichier
with open("Distribution angulaire\Data\mesures.txt", "r") as file:
    lines = file.readlines()

# Extraire les noms des colonnes depuis la première ligne
column_names = lines[0].strip().replace(',', '.').split()
lines = lines[1:]  # Sauter la première ligne (en-tête)

# Initialiser un compteur pour les sommes > 2
count = 0

# Correspondances entre combinaisons de colonnes et angles
column_to_angle = {
    tuple(sorted(['11', '21', '31'])): 0,
    tuple(sorted(['12', '22', '32'])): 0,
    tuple(sorted(['13', '23', '33'])): 0,
    tuple(sorted(['14', '24', '34'])): 0,
    tuple(sorted(['12', '23', '34'])): 0.989,
    tuple(sorted(['33', '22', '11'])): 0.989,
    tuple(sorted(['32', '21'])): 0.989,
    tuple(sorted(['24', '13'])): 0.989,
    tuple(sorted(['33', '24'])): -0.989,
    tuple(sorted(['14', '23', '32'])): -0.989,
    tuple(sorted(['13', '22', '31'])): -0.989,
    tuple(sorted(['12', '21'])): -0.989,
    tuple(sorted(['34', '13'])): 0.65,
    tuple(sorted(['33', '12'])): 0.65,
    tuple(sorted(['32', '11'])): 0.65,
    tuple(sorted(['33', '14'])): -0.65,
    tuple(sorted(['32', '13'])): -0.65,
    tuple(sorted(['31', '12'])): -0.65,
    tuple(sorted(['34', '22'])): 1.355,
    tuple(sorted(['33', '21'])): 1.355,
    tuple(sorted(['24', '12'])): 1.355,
    tuple(sorted(['23', '11'])): 1.355,
    tuple(sorted(['32', '24'])): -1.355,
    tuple(sorted(['31', '23'])): -1.355,
    tuple(sorted(['14', '22'])): -1.355,
    tuple(sorted(['13', '21'])): -1.355,
    tuple(sorted(['34', '21'])): 1.408,
    tuple(sorted(['11', '24'])): 1.408,
    tuple(sorted(['31', '24'])): -1.408,
    tuple(sorted(['21', '14'])): -1.408,
    
    # Ajouter d'autres combinaisons ici
}

# Initialiser un dictionnaire pour compter les occurrences des angles
angle_counts = {}
unknown_angle_count = 0  # Compteur pour les angles inconnus

# Parcourir chaque ligne
for line in lines:
    # Remplacer les virgules par des points et convertir les valeurs en une liste de nombres
    values = list(map(float, line.strip().replace(',', '.').split()))
    
    # Vérifier qu'il y a au moins 16 valeurs
    if len(values) >= 16:
        # Prendre les 16 dernières valeurs et calculer leur somme
        last_16_sum = sum(values[-16:])
        
        # Vérifier si la somme est supérieure à 2
        if last_16_sum >= 2:
            count += 1
            
            # Identifier les colonnes déclenchées (valeurs > 0 dans les 16 dernières)
            triggered_columns = [
                column_names[i] for i in range(len(values) - 15, len(values)) if values[i-1] > 0
            ]
            
            # Vérifier si une combinaison est déclenchée
            angle_found = False
            for combination, angle in column_to_angle.items():
                # Vérifier si toutes les colonnes de la combinaison sont dans les colonnes déclenchées
                if all(col in triggered_columns for col in combination):
                    angle = angle *180 / np.pi  # Convertir l'angle en degrés
                    print(f"Colonnes déclenchées : {triggered_columns}, Combinaison : {combination}, Angle : {angle}")
                    angle_counts[angle] = angle_counts.get(angle, 0) + 1
                    angle_found = True
                    break  # Une fois qu'une combinaison est trouvée, on arrête la recherche
            
            # Si aucune combinaison n'est trouvée, compter comme angle inconnu
            if not angle_found:
                unknown_angle_count += 1

# Afficher le résultat final
print(f"Nombre de sommes supérieures à 3 : {count}")
print(column_names)

# Diviser les comptes des angles spécifiques
for angle in angle_counts:
    if angle == 0:
        angle_counts[angle] /= 4  # Diviser par 4 pour l'angle 0
    elif angle == 0.989*180 / np.pi:
        angle_counts[angle] /= 4  # Diviser par 4 pour l'angle 0.989
    elif angle == 0.65*180 / np.pi:
        angle_counts[angle] /= 3  # Diviser par 3 pour l'angle 0.65
    elif angle == -0.989*180 / np.pi:
        angle_counts[angle] /= 4  # Diviser par 4 pour l'angle 0.989
    elif angle == -0.65*180 / np.pi:
        angle_counts[angle] /= 3  # Diviser par 3 pour l'angle 0.65
    elif angle == 1.355*180 / np.pi:
        angle_counts[angle] /= 4
    elif angle == -1.355*180 / np.pi:
        angle_counts[angle] /= 4
    elif angle == 1.408*180 / np.pi:
        angle_counts[angle] /= 2
    elif angle == -1.408*180 / np.pi:
        angle_counts[angle] /= 2

# Créer un histogramme des angles
angles = list(angle_counts.keys())
counts = list(angle_counts.values())

# Ajouter l'angle inconnu à 3.14
angles.append(180)
counts.append(unknown_angle_count)

# Ajustement avec la fonction cosinus carré
angles_array = np.array(angles[:-1])  # Exclure l'angle inconnu (180)
counts_array = np.array(counts[:-1])  # Exclure le compte inconnu
popt, pcov = curve_fit(cos_squared, angles_array, counts_array, p0=[1, 0, 0])
chi_squared = np.sum(((counts_array - cos_squared(angles_array, *popt)) ** 2) / counts_array)

# Calcul des erreurs de Poisson pour les counts, en tenant compte des divisions
errors = np.sqrt(np.array(counts))  # Erreurs de Poisson avant division
for i, angle in enumerate(angles[:-1]):  # Exclure l'angle inconnu
    if angle == 0:
        errors[i] /= 4  # Diviser par 4 pour l'angle 0
    elif angle == 0.989 * 180 / np.pi:
        errors[i] /= 4  # Diviser par 4 pour l'angle 0.989
    elif angle == 0.65 * 180 / np.pi:
        errors[i] /= 3  # Diviser par 3 pour l'angle 0.65
    elif angle == -0.989 * 180 / np.pi:
        errors[i] /= 4  # Diviser par 4 pour l'angle -0.989
    elif angle == -0.65 * 180 / np.pi:
        errors[i] /= 3  # Diviser par 3 pour l'angle -0.65
    elif angle == 1.355 * 180 / np.pi:
        errors[i] /= 4  # Diviser par 4 pour l'angle 1.355
    elif angle == -1.355 * 180 / np.pi:
        errors[i] /= 4  # Diviser par 4 pour l'angle -1.355
    elif angle == 1.408 * 180 / np.pi:
        errors[i] /= 2  # Diviser par 2 pour l'angle 1.408
    elif angle == -1.408 * 180 / np.pi:
        errors[i] /= 2  # Diviser par 2 pour l'angle -1.408

# Générer des données pour le fit
fit_angles = np.linspace(min(angles_array), max(angles_array), 500)
fit_counts = cos_squared(fit_angles, *popt)
colors = ['blue'] * (len(angles) - 1) + ['red']  # Bleu pour les angles connus, rouge pour l'inconnu

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(fit_angles, fit_counts, color='green', label="Fit: $a \cdot \cos^2(x + b) + c$")

# Ajouter les points avec des erreurs verticales (Poisson)
ax.errorbar(angles, counts, yerr=errors, fmt='o', color='blue', label="Données avec erreurs", zorder=3)

# Ajouter le fit au graphique
textstr = (
    r"$\bf{TREX}$"
    f"\n"
    r" Fit : $N = a\cos^2\left(x+b\right)+c$"
    f"\n"
    f"a = {popt[0]:.3f}\n"
    f"b = {popt[1]:.3f}\n"
    f"c = {popt[2]:.3f}\n"
    r"$\chi^2 =$ "f"{chi_squared:.3f}"
)
ax.text(
    0.95, 0.95, textstr,
    transform=ax.transAxes, fontsize=10,
    verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

# Configuration du graphique
ax.set_xlabel("Angles (degrés)")
ax.set_ylabel("Nombre de coups (N)")
ax.set_title("Distribution des angulaire des muons pour un systeme à 12 scintillateurs")
ax.legend()
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
print(f"Paramètres du fit : {popt}")
print(f"Covariance du fit : {pcov}")


# plt.plot(fit_angles, fit_counts, color='green', label="Fit: $a \cdot \cos^2(x + b) + c$")
# plt.legend()
# # Créer un histogramme avec des couleurs différentes
# colors = ['blue'] * (len(angles) - 1) + ['red']  # Bleu pour les angles connus, rouge pour l'inconnu
# plt.bar(angles, counts, color=colors, alpha=0.7, width=3)

# # Ajouter des labels et un titre
# plt.xlabel("Angles")
# plt.ylabel("Nombre d'occurrences")
# plt.title("Histogramme des angles attribués")
# plt.xticks(rotation=45)
# plt.tight_layout()

# # Afficher l'histogramme
# plt.show()