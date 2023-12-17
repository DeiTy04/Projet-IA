from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__, template_folder='C:/Users/18515/Downloads/Projet_IA/templates')

# Charger le modèle
model = load_model('./Model/model.h5')

def load_and_prepare_image(img_file):
    img = image.load_img(img_file, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Assurez-vous que le dossier temp existe
    temp_folder = 'temp'
    os.makedirs(temp_folder, exist_ok=True)

    if 'file' not in request.files:
        return jsonify(error='Aucun fichier trouvé'), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify(error='Aucun fichier sélectionné'), 400

    # Chemin pour la sauvegarde temporaire de l'image
    temp_image_path = os.path.join(temp_folder, file.filename)
    try:
        file.save(temp_image_path)
    except IOError as e:
        return jsonify(error=str(e)), 500

    try:
        img_array = load_and_prepare_image(temp_image_path)
        prediction = model.predict(img_array)
        probability = prediction[0][0]
        seuil = 0.5
        classe_predite = 'Chou-fleur' if probability < seuil else 'Tournesol'
    finally:
        # Supprime le fichier temporaire après utilisation
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

    return jsonify(classe_predite=classe_predite, probability=float(probability))

if __name__ == '__main__':
    app.run(debug=True)
