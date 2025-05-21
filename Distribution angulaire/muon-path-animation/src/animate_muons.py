import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

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
data_path = "Distribution angulaire\Data\mesures.txt"

# Extraction des trajectoires détectées
trajectories = []
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
            for combination in column_to_angle:
                if sorted(triggered) == list(combination):
                    # Ajoute la trajectoire (dans l'ordre du combination)
                    try:
                        traj = [detector_pos[label] for label in combination]
                        trajectories.append((combination, traj))
                    except KeyError:
                        pass
                    break

if not trajectories:
    print("Aucune trajectoire détectée dans les données.")
    exit()




# Préparation de la figure
fig, ax = plt.subplots(figsize=(10, 7))  # <-- Fenêtre plus grande
ax.set_xlim(-2, 8)  # <-- Décalage et zoom arrière
ax.set_ylim(-2, 6)
ax.set_aspect('equal')
ax.axis('off')



# Dessin des détecteurs
for label, (x, y) in detector_pos.items():
    circle = plt.Circle((x, y), 0.35, color='lightblue', ec='black', zorder=2)
    ax.add_patch(circle)
    ax.text(x, y, label, ha='center', va='center', fontsize=14, zorder=3)

muon_line, = ax.plot([], [], color = 'red', linewidth=3, markersize=12, label="Muon", zorder=4)
traj_text = ax.text(0, 4.5, "", fontsize=14, color="black")
frames_per_traj = 2

# Animation : chaque trajectoire est animée en le même temps
def init():
    muon_line.set_data([], [])
    traj_text.set_text("")
    return muon_line, traj_text

def animate(frame):
    traj_idx = frame // frames_per_traj
    step = frame % frames_per_traj

    if traj_idx >= len(trajectories):
        muon_line.set_data([], [])
        traj_text.set_text("")
        return muon_line, traj_text

    comb, traj = trajectories[traj_idx]
    n_points = len(traj)
    # Interpolation linéaire le long de la trajectoire
    if n_points == 1:
        x, y = traj[0]
    else:
        # Position fractionnaire le long du chemin
        pos = step / (frames_per_traj - 1) * (n_points - 1)
        idx = int(pos)
        t = pos - idx
        if idx >= n_points - 1:
            x, y = traj[-1]
        else:
            x = (1 - t) * traj[idx][0] + t * traj[idx + 1][0]
            y = (1 - t) * traj[idx][1] + t * traj[idx + 1][1]
    # Affiche la trajectoire jusqu'à la position courante
    path_x = [traj[0][0]]
    path_y = [traj[0][1]]
    for i in range(1, n_points):
        if i - 1 < pos:
            path_x.append(traj[i][0])
            path_y.append(traj[i][1])
        else:
            break
    # Ajoute la position courante interpolée
    if (x, y) != (path_x[-1], path_y[-1]):
        path_x.append(x)
        path_y.append(y)

    if len(path_x) > 1:
        # Calcul du vecteur directeur
        dx = path_x[-1] - path_x[0]
        dy = path_y[-1] - path_y[0]
        norm = np.hypot(dx, dy)
        if norm > 0:
            # Facteur d'étirement (ajuste à ta convenance)
            stretch = 1.2
            # Point de départ étiré
            x0 = path_x[0] - stretch * dx / norm
            y0 = path_y[0] - stretch * dy / norm
            # Point d'arrivée étiré
            x1 = path_x[-1] + stretch * dx / norm
            y1 = path_y[-1] + stretch * dy / norm
            path_x = [x0] + path_x[1:-1] + [x1]
            path_y = [y0] + path_y[1:-1] + [y1]

    muon_line.set_data(path_x, path_y)
    traj_text.set_text(f"Trajectoire détectée : {comb}")
    return muon_line, traj_text

total_frames = len(trajectories) * frames_per_traj
ani = animation.FuncAnimation(
    fig, animate, frames=total_frames, init_func=init,
    blit=True, interval=80, repeat=False
)

plt.title("Trajectoires réelles des muons détectées")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()