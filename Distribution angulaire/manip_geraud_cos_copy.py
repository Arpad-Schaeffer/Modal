import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.ticker import MultipleLocator, AutoLocator, AutoMinorLocator
import os

# -----------------------------------------
#dans ce doc j'ai essaye de retranscrire une modelisation plus juste en cos(a*x) pour avoir des DL plus juste et un chi suared plus petit.
#mais jai pas l'impression que ca marche
# -----------------------------------------

save_fig = False

# Fonction cosinus carré pour le fit
def cos_squared(x, a, b,c):
    return a * (np.cos(c*np.radians(x)) ** 2) + b

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
print(angles)
print(counts)
# Ajouter l'angle inconnu à 3.14
#angles.append(180)
#counts.append(unknown_angle_count)



# Calcul des erreurs de Poisson pour les counts, en tenant compte des divisions
errors = np.sqrt(np.array(counts))  # Erreurs de Poisson avant division
for i, angle in enumerate(angles):  # Exclure l'angle inconnu
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

# Erreurs sur les angles (en, ordre croissant en valeur absolue)
errors_x = [2.98,4,2.92,2.92,2.98,0.46,0.22,0.22,0.46]  # Correspondance mise à jour pour les angles en degrés
print(errors_x)

# Ajustement avec la fonction cosinus carré
angles_array = np.array(angles)  # Exclure l'angle inconnu (180)
counts_array = np.array(counts)  # Exclure le compte inconnu
popt, pcov = curve_fit(cos_squared, angles_array, counts_array, p0=[1500, 0,1.29])
chi_squared = np.sum(((counts_array - cos_squared(angles_array, *popt)) ** 2) / errors**2)
dof = len(counts_array) - len(popt)  # Degrés de liberté
chi_squared_reduced = chi_squared / dof



# Générer des données pour le fit
fit_angles = np.linspace(min(angles_array), max(angles_array), 500)
fit_counts = cos_squared(fit_angles, *popt)
colors = ['blue'] * (len(angles) - 1) + ['red']  # Bleu pour les angles connus, rouge pour l'inconnu

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(fit_angles, fit_counts, color='k', label="Fit: $a \cdot \cos^2(cx) + b$",alpha=0.7)

# Ajouter les points avec erreurs verticales ET horizontales
ax.errorbar(angles, counts, xerr=errors_x, yerr=errors, fmt='o', color='C1', label="Données avec erreurs", zorder=3,alpha=0.7, capsize=3)

# Ajouter le fit au graphique
textstr = (
    r"$\bf{TREX}$"
    f"\n"
    r" Fit : $N = a\cos^2\left(cx\right)+b$"
    f"\n"
    f"a = {popt[0]:.3f}\n"
    f"b = {popt[1]:.3f}\n"
    f"c = {popt[2]:.3f}\n"
    r"$\chi^2_{red} =$ "f"{chi_squared_reduced:.3f}"
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

if save_fig:
    os.makedirs("Latex/Images", exist_ok=True)
    fig.savefig("Latex/Images/cos_angles.png", dpi=300, bbox_inches='tight')
    print("Graphique sauvegardé dans : Latex/Images/cos_angles.png")

plt.show()
print(f"Paramètres du fit : {popt}")
print(f"Covariance du fit : {pcov}")

#-------------------------------------------------------------
#test de lapporx de cos^2 pour les angles plus petits
#-------------------------------------------------------------
# Trier les angles et les comptes associés
sorted_indices = np.argsort(angles_array)
angles_array_sorted = angles_array[sorted_indices]
counts_array_sorted = counts_array[sorted_indices]
errors_sorted = errors[sorted_indices]

# Exclure les deux angles les plus négatifs et les deux angles les plus positifs
angles_trimmed = angles_array_sorted[2:-2]
counts_trimmed = counts_array_sorted[2:-2]
errors_trimmed = errors_sorted[2:-2]
errors_x_trimmed = np.array([2.92,2.98,4,2.98,2.92])  # Erreurs sur les angles


# Refaire le fit avec les données réduites
popt_trimmed, pcov_trimmed = curve_fit(cos_squared, angles_trimmed, counts_trimmed, p0=[1500, 0,1])

# Recalculer le chi-squared avec les données réduites
chi_squared_trimmed = np.sum(((counts_trimmed - cos_squared(angles_trimmed, *popt_trimmed)) ** 2) / errors_trimmed**2)
dof_trimmed = len(counts_trimmed) - len(popt_trimmed)  # Degrés de liberté
chi_squared_reduced_trimmed = chi_squared_trimmed / dof_trimmed

print(f"Paramètres du fit (données réduites) : {popt_trimmed}")
print(f"Chi-squared réduit après exclusion : {chi_squared_reduced_trimmed:.3f}")

# Générer des données pour le nouveau fit
fit_angles_trimmed = np.linspace(min(angles_trimmed), max(angles_trimmed), 500)
fit_counts_trimmed = cos_squared(fit_angles_trimmed, *popt_trimmed)

# Tracer le nouveau fit
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(fit_angles_trimmed, fit_counts_trimmed, color='k', label="Fit (données réduites): $a \cdot \cos^2(cx) + b$", alpha=0.7)
ax.errorbar(angles_trimmed, counts_trimmed, xerr=errors_x_trimmed, yerr=errors_trimmed, fmt='o', color='C1', label="Données réduites avec erreurs", zorder=3, alpha=0.7, capsize=3)

# Ajouter le texte du fit
textstr_trimmed = (
    r"$\bf{TREX}$"
    f"\n"
    r" Fit : $N = a\cos^2\left(cx\right)+b$"
    f"\n"
    f"a = {popt_trimmed[0]:.3f}\n"
    f"b = {popt_trimmed[1]:.3f}\n"
    f"c = {popt_trimmed[2]:.3f}\n"
    r"$\chi^2_{red} =$ "f"{chi_squared_reduced_trimmed:.3f}"
)
ax.text(
    0.95, 0.95, textstr_trimmed,
    transform=ax.transAxes, fontsize=10,
    verticalalignment='top', horizontalalignment='right',
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

# Configuration du graphique
ax.set_xlabel("Angles (degrés)")
ax.set_ylabel("Nombre de coups (N)")
ax.set_title("Fit après exclusion des angles extrêmes")
ax.legend(fontsize=11, loc='upper left', frameon=False, bbox_to_anchor=(0.02, 0.98), borderaxespad=1)
ax.grid(False)

plt.tight_layout()
plt.show()


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