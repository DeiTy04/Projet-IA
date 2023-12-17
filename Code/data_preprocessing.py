from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
import os
import glob

# Configuration de l'augmentation des données
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Chargement de toutes les images depuis un dossier
def charger_toutes_les_images_du_dossier(dossier, target_size=(128, 128)):
    images = []
    # Charger les images JPG et PNG
    for filepath in glob.glob(os.path.join(dossier, '*.jpg')) + glob.glob(os.path.join(dossier, '*.jpeg')) + glob.glob(os.path.join(dossier, '*.png')):
        img = load_img(filepath, target_size=target_size)
        img = img_to_array(img)
        images.append(img)
    return np.array(images)

# Chemin des dossiers d'images
chemin_tournesols = './Data/Tournesols'  # Images JPG
chemin_chou_fleurs = './Data/Chou-fleurs'  # Images PNG

# Charger les images
images_tournesols = charger_toutes_les_images_du_dossier(chemin_tournesols)
images_chou_fleurs = charger_toutes_les_images_du_dossier(chemin_chou_fleurs)

# Fusionner les images
toutes_les_images = np.concatenate([images_tournesols, images_chou_fleurs], axis=0)

# Appliquer l'augmentation des données
images_augmentees = []
for lot_X in datagen.flow(toutes_les_images, batch_size=32):
    images_augmentees.append(lot_X)
    if len(images_augmentees) * 32 >= toutes_les_images.shape[0]:
        break

images_augmentees = np.concatenate(images_augmentees, axis=0)

# Afficher le nombre d'images augmentées et leurs dimensions
print("Nombre d'images augmentées:", images_augmentees.shape[0])
print("Dimensions des images:", images_augmentees.shape[1:])

# Créer les étiquettes (en supposant que les tournesols sont étiquetés comme 1 et les choux-fleurs comme 0)
etiquettes = np.concatenate([np.ones(images_tournesols.shape[0]), np.zeros(images_chou_fleurs.shape[0])])

# Enregistrer les données
np.save('./Model/images_augmentees.npy', images_augmentees)
np.save('./Model/etiquettes.npy', etiquettes)
