import pyvisa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re

# Définissez cette constante sur True pour enregistrer les données
SAVE_DATA = True

def get_channel_data(scope, channel):
    """
    Configure le canal, récupère le préambule et la courbe de données.
    Retourne un tuple contenant :
      - les points de données (numpy array),
      - la chaîne du préambule,
      - l'info du canal extraite (ex. "ChX, DC coupling, ..."),
      - le nombre de points.
    """
    # Configuration du canal
    scope.write(f"DATA:SOURCE {channel}")
    scope.write("DATA:WIDTH 1")
    scope.write("DATA:ENCdg ASCii")
    
    # Récupération du préambule
    preamble = scope.query("WFMPRE?")
    preamble_fields = preamble.split(';')
    
    # Extraction de l'info du canal (champ 7) avec vérification de la longueur
    if len(preamble_fields) > 6:
        channel_info = preamble_fields[6].strip('"')
    else:
        channel_info = "Information indisponible"
    
    # Extraction du nombre de points (champ 6)
    try:
        num_points = int(preamble_fields[5])
    except Exception:
        num_points = None
    
    # Récupération des données de la courbe
    data_str = scope.query("CURVE?")
    try:
        data_points = np.array(data_str.strip().split(','), dtype=float)
    except Exception as e:
        print(f"Erreur lors de la conversion des données pour {channel}: {e}")
        data_points = np.array([])
    
    return data_points, preamble, channel_info, num_points

# Crée le gestionnaire de ressources VISA
rm = pyvisa.ResourceManager()

# Affiche les ressources disponibles
resources = rm.list_resources()
print("Ressources disponibles :", resources)

# Chaîne de ressource pour l'oscilloscope Tektronix
resource_str = 'USB0::0x0699::0x03A6::C057600::INSTR'
scope = rm.open_resource(resource_str)

# Affiche l'identification de l'instrument
idn_response = scope.query("*IDN?")
print("Identification de l'instrument :", idn_response)

# Délai pour que l'acquisition soit prête
scope.timeout = 15000

# Liste des canaux à tester
channels = ["CH1", "CH2", "CH3", "CH4"]
active_channels = {}

# Pour chaque canal, on tente de récupérer ses données.
for ch in channels:
    try:
        data, preamble, info, num_points = get_channel_data(scope, ch)
        # On considère le canal actif s'il renvoie au moins quelques points
        if len(data) > 0:
            active_channels[ch] = {"data": data, "preamble": preamble, "info": info, "num_points": num_points}
            print(f"{ch} est actif avec {len(data)} points récupérés.")
        else:
            print(f"{ch} n'est pas utilisé (aucune donnée récupérée).")
    except pyvisa.errors.VisaIOError as e:
        print(f"{ch} n'est pas utilisé (erreur: {e}).")

if not active_channels:
    print("Aucun canal actif détecté.")
    exit()

# Utilisation du premier canal actif pour construire le vecteur temps
first_active_channel = next(iter(active_channels))
first_data = active_channels[first_active_channel]
info_first = first_data["info"]
num_points = first_data["num_points"]

# Extraction du réglage horizontal (temps/div) depuis l'info du premier canal actif
match_time = re.search(r'([\d.Ee+-]+)\s*s/div', info_first)
if match_time and num_points:
    time_per_div = float(match_time.group(1))
    # On suppose ici 12 divisions horizontales
    total_time = time_per_div * 12
    time_vector = np.linspace(0, total_time, num_points)
    x_label = "Temps (s)"
else:
    time_vector = np.arange(len(first_data["data"]))
    x_label = "Index (points)"

y_label = "Tension (V)"

# Tracé des données des canaux actifs sur le même graphique
plt.figure(figsize=(10, 6))
for ch, vals in active_channels.items():
    # On peut insérer un saut de ligne dans le label si souhaité, ici chaque entrée sera sur une ligne
    label_text = f"{ch}: {vals['info']}"
    plt.plot(time_vector, vals["data"], label=label_text)
plt.title("Acquisition Multi-Canaux - Tektronix TDS 2024C", fontsize=16)
plt.xlabel(x_label, fontsize=14)
plt.ylabel(y_label, fontsize=14)
plt.grid(True)

# Positionnement de la légende en dessous du graphique, en une seule colonne (une ligne par canal)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), fontsize=12, ncol=1)

# Enregistrement des données dans un fichier CSV si SAVE_DATA est True
if SAVE_DATA:
    data_dict = {x_label: time_vector}
    for ch, vals in active_channels.items():
        data_dict[f"{ch} (V)"] = vals["data"]
    df = pd.DataFrame(data_dict)
    
    # Création d'une chaîne de caractères pour la date et l'heure courante au format "jour-mois-année_heure-minute"
    import datetime
    date_str = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M")
    
    # Création d'une chaîne listant les canaux utilisés séparés par un underscore
    channels_used = "_".join(active_channels.keys())
    
    # Création du nom de fichier avec la date et les canaux utilisés
    filename = f"/Users/malotamalet/Desktop/2A/Modal/Modal/Temps de vie/Data/Data_{date_str}_{channels_used}.csv"

    df.to_csv(filename, index=False)
    print(f"Données enregistrées dans {filename}")

    # Enregistrement du plot dans le même dossier
    plot_filename = f"/Users/malotamalet/Desktop/2A/Modal/Modal/Temps de vie/Data/Plot_{date_str}_{channels_used}.png"
    plt.savefig(plot_filename, bbox_inches='tight')
    print(f"Plot enregistré dans {plot_filename}")

    plt.show()