import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import scipy.stats as st

fig, ax = plt.subplots(figsize=(10,6))

# -----------------------------
# Lecture des données
# -----------------------------
filename = "Distribution angulaire/Data/angles.csv"  # Remplacez par le chemin réel
skip_lines = 20  # Nombre de lignes à ignorer au début du fichier
try:
    data = pd.read_csv(filename, skiprows=skip_lines)
except Exception as e:
    print(f"Erreur lors de la lecture de {filename}: {e}")
    exit(1)

L32 = data['L32']
L21 = data['L21']
H = 48

angles = np.arctan((L32 + L21) / H) * 180 / np.pi  # en degrés
count = data['coincidences']                                # sur 10 min
count_random = data['coincidences_fortuites']               # sur 10 min
eff_count = count / np.cos(np.deg2rad(angles))                 # sur 10 min

# Passage en compte/minute
count_min       = count / 10
count_random_min   = count_random / 10
eff_count_min = eff_count / 10
count_diff_min     = (count - count_random) / 10
ratio = count_diff_min / count_random_min

# Erreurs Poisson
# Incertitudes sur L32 et L21 (à ajuster selon vos mesures)
err_L32 = 1
err_L21 = 1  
dtheta_dL = (180/np.pi) * (1/H) / (1 + ((18 + L32 + L21)/H)**2)
err_angles = dtheta_dL * np.sqrt(err_L32**2 + err_L21**2)
err_count     = np.sqrt(count) / 10
err_random= np.sqrt(count_random) / 10
err_eff_count = np.sqrt(
    err_count**2 +
    (err_angles * np.sin(np.deg2rad(angles)))**2
)
err_diff   = np.sqrt(err_count**2 + err_random**2) / 10
err_ratio = np.sqrt((err_diff/count_diff_min)**2 + (err_random/ratio)**2)

# grille fine pour la courbe
x_fit = np.linspace(0, angles.max(), 500)

# Fit en cos²
def cos2(theta, A, offset):
    return A * np.cos(np.deg2rad(theta))**2 + offset

# estimation initiale des paramètres
p0 = [count_min.max(), count_min.min()]

params, cov = curve_fit(
    cos2, angles, eff_count_min,
    sigma=err_eff_count, absolute_sigma=True, p0=p0
)

# calcul des incertitudes et des p‑values
se = np.sqrt(np.diag(cov))
dof = len(angles) - len(params)
t_vals = params / se
p_vals = 2 * (1 - st.t.cdf(np.abs(t_vals), dof))

y_fit = cos2(x_fit, *params)
ax.plot(
    x_fit, y_fit, '-', color='red',
    label=(
        f'Fit cos²: A={params[0]:.1f}±{se[0]:.1f} (p={p_vals[0]:.2f}), '
        f'offset={params[1]:.1f}±{se[1]:.1f} (p={p_vals[1]:.2f})'
    )
)

# -----------------------------
# Construction du figure 
# -----------------------------
ax.errorbar(
    angles, eff_count_min, yerr=err_eff_count, xerr=err_angles,
    fmt='o', ms=5, capsize=3, label='Coïncidences corrigées'
)
# tracé des données brutes
ax.errorbar(
    angles, count_min, yerr=err_count, xerr=err_angles,
    fmt='s', ms=5, capsize=3, label='Coïncidences brutes'
)
    
ax.set_ylabel("Nombre de coïncidences par minute")
ax.set_xlabel("θ (degrés)")
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, which='both', ls='--', alpha=0.3)

xmin, xmax = angles.min()*0.8, angles.max()*1.2
ymin, ymax = None, None  # ou définir manuellement

plt.tight_layout(h_pad=0.1)
plt.show()