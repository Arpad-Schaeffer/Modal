import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import curve_fit

# Détecteurs et leur position (espacés)
detector_pos = {
    "11": (6, 0), "12": (4, 0), "13": (2, 0), "14": (0, 0),
    "21": (6, 2), "22": (4, 2), "23": (2, 2), "24": (0, 2),
    "31": (6, 4), "32": (4, 4), "33": (2, 4), "34": (0, 4)
}

# Combinaisons d'angles comme dans manip_geraud.py
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
}

# Chemin vers le fichier mesures.txt (adapter si besoin)
data_path = "Distribution angulaire/Data/mesures.txt"

# Extraction des trajectoires détectées
trajectories = []
angles_detected = []
with open(data_path, "r") as file:
    lines = file.readlines()
column_names = lines[0].strip().replace(',', '.').split()
lines = lines[1:]

for line in lines:
    values = list(map(float, line.strip().replace(',', '.').split()))
    if len(values) >= 16:
        last_16_sum = sum(values[-16:])
        if last_16_sum >= 2:
            triggered = [
                column_names[i] for i in range(len(values) - 15, len(values))
                if values[i-1] > 0
            ]
            # On cherche une combinaison exacte
            for combination, angle in column_to_angle.items():
                if sorted(triggered) == list(combination):
                    try:
                        traj = [detector_pos[label] for label in combination]
                        trajectories.append((combination, traj))
                        angles_detected.append(angle * 180 / np.pi)  # Convertir en degrés
                    except KeyError:
                        pass
                    break

if not trajectories:
    print("Aucune trajectoire détectée dans les données.")
    exit()

# Fonction cos² pour le fit
def cos_squared(x, a, b, c):
    return a * (np.cos(np.radians(x) + b) ** 2) + c

# Calcul des occurrences des angles
angle_counts = {}
for angle in angles_detected:
    angle_counts[angle] = angle_counts.get(angle, 0) + 1

# Ajouter une fonction pour ajuster les poids des angles
def adjust_angle_weights(angle_counts):
    adjusted_counts = {}
    for angle, count in angle_counts.items():
        if angle == 0:
            adjusted_counts[angle] = count / 4
        elif angle == 0.989 * 180 / np.pi:
            adjusted_counts[angle] = count / 4
        elif angle == 0.65 * 180 / np.pi:
            adjusted_counts[angle] = count / 3
        elif angle == -0.989 * 180 / np.pi:
            adjusted_counts[angle] = count / 4
        elif angle == -0.65 * 180 / np.pi:
            adjusted_counts[angle] = count / 3
        elif angle == 1.355 * 180 / np.pi:
            adjusted_counts[angle] = count / 4
        elif angle == -1.355 * 180 / np.pi:
            adjusted_counts[angle] = count / 4
        elif angle == 1.408 * 180 / np.pi:
            adjusted_counts[angle] = count / 2
        elif angle == -1.408 * 180 / np.pi:
            adjusted_counts[angle] = count / 2
        else:
            adjusted_counts[angle] = count  # Pas de division pour les autres angles
    return adjusted_counts

# Préparation des données pour le fit
angles = list(angle_counts.keys())
counts = list(angle_counts.values())
angles_array = np.array(angles)
counts_array = np.array(counts)

# Ajustement avec la fonction cos²
popt, pcov = curve_fit(cos_squared, angles_array, counts_array, p0=[1, 0, 0])

# Préparation de la figure avec deux sous-graphiques
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

# Sous-graphe 1 : Animation
ax1.set_xlim(-2, 8)
ax1.set_ylim(-2, 6)
ax1.set_aspect('equal')
ax1.axis('off')

# Dessin des détecteurs
for label, (x, y) in detector_pos.items():
    circle = plt.Circle((x, y), 0.35, color='lightblue', ec='black', zorder=2)
    ax1.add_patch(circle)
    ax1.text(x, y, label, ha='center', va='center', fontsize=14, zorder=3)

muon_line, = ax1.plot([], [], color='red', linewidth=3, markersize=12, label="Muon", zorder=4)
traj_text = ax1.text(0, 4.5, "", fontsize=14, color="black")
frames_per_traj = 2

# Initialisation des données pour l'histogramme progressif
current_angles = []  # Angles détectés au fil de l'animation
current_angle_counts = {}  # Comptage progressif des angles

# Initialisation de l'histogramme et du fit
bar_plot = None
fit_line = None
fit_params_text = None
detection_count_text = None  # <-- Ajout

def init():
    muon_line.set_data([], [])
    traj_text.set_text("")
    global bar_plot, fit_line, fit_params_text, detection_count_text
    bar_plot = ax2.bar([], [], color='blue', alpha=0.7, label="Occurrences")
    fit_line, = ax2.plot([], [], color='green', label="Fit: $a \cdot \cos^2(x + b) + c$")
    fit_params_text = ax2.text(0.05, 0.95, "", transform=ax2.transAxes, va='top', fontsize=12, color='black', bbox=dict(facecolor='white', alpha=0.7))
    detection_count_text = ax2.text(0.95, 0.95, "", transform=ax2.transAxes, va='top', ha='right', fontsize=12, color='purple', bbox=dict(facecolor='white', alpha=0.7))  # <-- Ajout
    return muon_line, traj_text, *bar_plot, fit_line, fit_params_text, detection_count_text  # <-- Ajout

# Modifier la fonction animate pour inclure l'ajustement des poids
def animate(frame):
    global current_angles, current_angle_counts, bar_plot, fit_line, fit_params_text, detection_count_text

    traj_idx = frame // frames_per_traj
    step = frame % frames_per_traj

    if traj_idx >= len(trajectories):
        return muon_line, traj_text, *bar_plot, fit_line, fit_params_text, detection_count_text  # <-- Ajout

    # Trajectoire actuelle
    comb, traj = trajectories[traj_idx]
    n_points = len(traj)
    if n_points == 1:
        x, y = traj[0]
    else:
        pos = step / (frames_per_traj - 1) * (n_points - 1)
        idx = int(pos)
        t = pos - idx
        if idx >= n_points - 1:
            x, y = traj[-1]
        else:
            x = (1 - t) * traj[idx][0] + t * traj[idx + 1][0]
            y = (1 - t) * traj[idx][1] + t * traj[idx + 1][1]
    path_x = [traj[0][0]]
    path_y = [traj[0][1]]
    for i in range(1, n_points):
        if i - 1 < pos:
            path_x.append(traj[i][0])
            path_y.append(traj[i][1])
        else:
            break
    if (x, y) != (path_x[-1], path_y[-1]):
        path_x.append(x)
        path_y.append(y)

    muon_line.set_data(path_x, path_y)
    traj_text.set_text(f"Trajectoire détectée : {comb}")

    # Mise à jour des angles détectés
    if step == 0:  # Ajouter un nouvel angle au début de chaque trajectoire
        angle = angles_detected[traj_idx]
        current_angles.append(angle)
        current_angle_counts[angle] = current_angle_counts.get(angle, 0) + 1

        # Ajuster les poids des angles
        adjusted_counts = adjust_angle_weights(current_angle_counts)

        # Mise à jour de l'histogramme
        angles = list(adjusted_counts.keys())
        counts = list(adjusted_counts.values())
        angles_array = np.array(angles)
        counts_array = np.array(counts)

        # Recalcul du fit
        if len(angles) > 2:  # Nécessaire pour effectuer un fit
            popt, _ = curve_fit(cos_squared, angles_array, counts_array, p0=[1, 0, 0])
            fit_angles = np.linspace(min(angles_array), max(angles_array), 500)
            fit_counts = cos_squared(fit_angles, *popt)

            # Mise à jour du graphique
            for bar in bar_plot:
                bar.remove()
            bar_plot = ax2.bar(angles, counts, color='blue', alpha=0.7, label="Occurrences")
            fit_line.set_data(fit_angles, fit_counts)
            ax2.legend([
                "Occurrences",
                r"Fit : $a \cos^2(x + b) + c$\n$a$=amplitude, $b$=décalage, $c$=offset"
            ])

            # Affichage des paramètres du fit
            fit_params_text.set_text(
                f"a = {popt[0]:.2f}\nb = {popt[1]:.2f}\nc = {popt[2]:.2f}\n"
                r"Fit : $a \cos^2(x + b) + c$"
            )
        else:
            fit_params_text.set_text("")

        # Affichage du nombre de détections
        detection_count_text.set_text(f"Détections : {len(current_angles)}")  # <-- Ajout

    return muon_line, traj_text, *bar_plot, fit_line, fit_params_text, detection_count_text  # <-- Ajout

# Animation
total_frames = len(trajectories) * frames_per_traj

ani = animation.FuncAnimation(
    fig, animate, frames=total_frames, init_func=init,
    blit=True, interval=10, repeat=False
)

plt.tight_layout()
plt.show()