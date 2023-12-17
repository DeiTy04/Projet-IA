# predict.py

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Charger le modèle
model = load_model('./Model/model.h5')

# Charger et prétraiter l'image
def load_and_prepare_image(image_path):
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Prédiction
# Assurez-vous que le chemin ici pointe vers l'image que vous souhaitez prédire
image_path = './la-voiture-noire-de-bugatti-modele-unique-photo-dr-1608828241.jpg'  # Modifiez avec le chemin de votre image
img_array = load_and_prepare_image(image_path)
prediction = model.predict(img_array)

# Obtenir la probabilité de sortie du modèle
probability = prediction[0][0]

# Définir un seuil
seuil = 0.6
seuilChou = 0.19

def comparaisonImage(probability, seuil) :
    if probability < seuilChou :
        return 'Chou-fleur'
    elif probability <= seuil :
        return 'Tournesol'
    else:
        return 'Ni Chou-fleur et Ni Tournesol'


classe_predite = comparaisonImage(probability, seuil)

print(f"Résultat de la prédiction : {classe_predite} (Probabilité : {probability})")