'''
Descrição: O script carrega e pré-processa imagens RGB e suas máscaras segmentadas, 
           redimensionando-as e normalizando os valores dos pixels. As imagens e 
           máscaras são divididas em conjuntos de treinamento e validação.
Data:      05/08/2024
'''

# Importação das bibliotecas
import os  # Manipulação de arquivos e diretórios
import numpy as np # Operações numéricas
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # Funções para carregar e converter imagens
from sklearn.model_selection import train_test_split  # Função para dividir dados em conjuntos de treinamento e validação

# Função que carrega e pré-processa as imagens RGB e segmentadas.
def load_data(rgb_dir, segmented_dir, img_size=(256, 256)):
   
    images = [] # Lista para armazenar as imagens RGB
    masks = [] # Lista para armazenar as máscaras segmentadas

    # Itera sobre as imagens no diretório RGB
    for filename in os.listdir(rgb_dir):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            rgb_path = os.path.join(rgb_dir, filename) # Caminho completo da imagem RGB
            segmented_path = os.path.join(segmented_dir, filename) # Caminho completo da máscara segmentada

            # Carrega e redimensiona a imagem RGB
            rgb_img = load_img(rgb_path, target_size=img_size) # Carrega a imagem e redimensiona
            rgb_array = img_to_array(rgb_img) / 255.0 # Converte a imagem para array e normaliza os valores
            images.append(rgb_array) # Adiciona a imagem à lista

            # Carrega e redimensiona a máscara segmentada
            segmented_img = load_img(segmented_path, target_size=img_size, color_mode="grayscale") # Carrega a máscara em escala de cinza
            segmented_array = img_to_array(segmented_img) / 255.0 # Converte a máscara para array e normaliza os valores
            masks.append(segmented_array) # Adiciona a máscara à lista


    # Converte listas para arrays NumPy
    images = np.array(images) # Array com todas as imagens RGB
    masks = np.array(masks) # Array com todas as máscaras segmentadas

    # Divide os dados em conjuntos de treinamento e validação
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)
    
    return X_train, X_val, y_train, y_val
