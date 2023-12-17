import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Charger les images augmentées précédemment enregistrées et les étiquettes
images_augmentees = np.load('./Model/images_augmentees.npy')
etiquettes = np.load('./Model/etiquettes.npy')

# Diviser les données en ensemble d'entraînement et ensemble de test
X_train, X_test, y_train, y_test = train_test_split(images_augmentees, etiquettes, test_size=0.2, random_state=42)

# Construire le modèle
modele = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

modele.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
historique = modele.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))

# Enregistrer le modèle
modele.save('./Model/model.h5')

# Tracer l'évolution de l'exactitude et de la perte pendant l'entraînement
exactitude = historique.history['accuracy']
exactitude_val = historique.history['val_accuracy']
perte = historique.history['loss']
perte_val = historique.history['val_loss']

epochs = range(len(exactitude))

# Tracer la courbe d'exactitude d'entraînement et de validation
plt.plot(epochs, exactitude, 'r', label="Exactitude d'entraînement")
plt.plot(epochs, exactitude_val, 'b', label="Exactitude de validation")
plt.title('Courbe d\'exactitude d\'entraînement et de validation')
plt.xlabel('Époque')
plt.ylabel('Exactitude')
plt.legend()
plt.show()

plt.figure()

# Tracer la courbe de perte d'entraînement et de validation
plt.plot(epochs, perte, 'r', label='Perte d\'entraînement')
plt.plot(epochs, perte_val, 'b', label='Perte de validation')
plt.title('Courbe de perte d\'entraînement et de validation')
plt.xlabel('Époque')
plt.ylabel('Perte')
plt.legend()
plt.show()
