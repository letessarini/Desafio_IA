'''
Descrição: Este script realiza a segmentação de imagens usando um modelo treinado.
           Carrega as imagens RGB de um diretório especificado, aplica o modelo para 
           gerar a segmentação, e salva as imagens segmentadas em um diretório de saída 
           indicado.
Data:      05/08/2024
'''

import argparse # Argumentos na linha de comando
import os # Manipulação de arquivos e diretórios
import numpy as np # Operações numéricas
import tensorflow as tf  # Para carregar o modelo e fazer previsões
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img # Para manipular imagens

# Configuração dos argumentos de comando
parser = argparse.ArgumentParser(description='Inferência do modelo para segmentação de imagens.')
parser.add_argument('--rgb', type=str, required=True, help='Caminho para o diretório contendo as imagens RGB a serem segmentadas.')
parser.add_argument('--modelpath', type=str, required=True, help='Caminho do modelo treinado.')
parser.add_argument('--output', type=str, required=True, help='Caminho para o diretório onde as imagens segmentadas serão salvas.')

# Análise dos argumentos fornecidos no terminal
args = parser.parse_args()

# Carregar o modelo treinado
model = tf.keras.models.load_model(args.modelpath)

# Verifica se o diretório de saída existe, se não, cria
if not os.path.exists(args.output):
    os.makedirs(args.output)

# Lista todos os arquivos no diretório de entrada
for filename in os.listdir(args.rgb):
    # Verifica se o arquivo é uma imagem
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
        # Caminho completo da imagem
        image_path = os.path.join(args.rgb, filename)
        
        # Carregar a imagem RGB
        image = load_img(image_path, target_size=(256, 256))  # Redimensiona a imagem para o tamanho esperado pelo modelo
        image_array = img_to_array(image) / 255.0  # Converte a imagem para um array NumPy e normaliza

        # Adiciona uma dimensão extra para o batch
        image_array = np.expand_dims(image_array, axis=0)

        # Realizar a previsão
        prediction = model.predict(image_array)

        # Remove a dimensão do batch
        prediction = np.squeeze(prediction)

        # Verifica a forma da previsão e adiciona a dimensão de canal se necessário
        if prediction.ndim == 2:
            prediction = np.expand_dims(prediction, axis=-1)

        # Converte a previsão para uma imagem
        segmented_image = array_to_img(prediction * 255.0)  # Multiplica por 255 para reverter a normalização

        # Cria o caminho completo para salvar a imagem segmentada
        output_path = os.path.join(args.output, filename)

        # Verifica e adiciona uma extensão padrão se a extensão não for fornecida
        if not os.path.splitext(output_path)[1]:
            output_path += '.png'

        # Salva a imagem segmentada
        segmented_image.save(output_path)
        print(f"Imagem segmentada salva em {output_path}")
