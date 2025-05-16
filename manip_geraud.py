# Ouvrir et lire le fichier
with open("mesures_arpad.txt", "r") as file:
    lines = file.readlines()

# Extraire les noms des colonnes depuis la première ligne
column_names = lines[0].strip().replace(',', '.').split()
lines = lines[1:]  # Sauter la première ligne (en-tête)

# Initialiser un compteur pour les sommes > 2
count = 0

# Correspondances entre combinaisons de colonnes et angles
column_to_angle = {
    ('11', '21','31'): 0,
    ('12', '22','32'): 0,
    ('13', '23','33'): 0,
    ('14', '24','34'): 0,
    ('12','23','34'): 0.989,
    ('33','22','11'): 0.989,
    ('32','21'): 0.989,
    ('24','13'): 0.989,
    ('33','24'): - 0.989,
    ('14','23','32'):-0.989,
    ('13','22','31'):-0.989,
    ('12','21'):-0.989,
    
    
    # Ajouter d'autres combinaisons ici
}

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
                column_names[i] for i in range(len(values) - 16, len(values)) if values[i] > 0
            ]
            
            # Trouver l'angle correspondant à la combinaison de colonnes
            triggered_columns_tuple = tuple(triggered_columns)
            angle = column_to_angle.get(triggered_columns_tuple, "Inconnu")
            
            # Afficher la ligne, les colonnes déclenchées et l'angle
            print(f"Colonnes déclenchées : {triggered_columns}, Angle : {angle}")

# Afficher le résultat final
print(f"Nombre de sommes supérieures à 2 : {count}")