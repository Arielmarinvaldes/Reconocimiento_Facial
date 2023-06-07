import cv2
import os
import numpy as np


# carpeta donde deve estar el modelo reconocido de los rostros guardados
data_path = 'Data_Face'
people = os.listdir(data_path)
print(people)
labels = []
faces_data = []
label = 0

# leer imagen por imagen 
for person in people:
    person_path = data_path + '/' + person
    print('Leyendo las imágenes')
    for file_name in os.listdir(person_path):
        print('Faces: ', person + '/' + file_name)
        labels.append(label)
        faces_data.append(cv2.imread(person_path + '/' + file_name, 0))
    label += 1

# entrenado modelo 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
print("Entrenando...")
face_recognizer.train(faces_data, np.array(labels))

# Almacenando el modelo
face_recognizer.write('LBPHFaceModel.xml')
print('Modelo almacenado!')