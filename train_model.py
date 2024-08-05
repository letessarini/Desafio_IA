'''
Descrição: Ccript que configura e treina o modelo para segmentação de imagens.
Data:      05/08/2024
'''

# Importação das bibliotecas
import argparse
import os
from data_preparation import load_data
from model import unet_model
import tensorflow as tf
import matplotlib.pyplot as plt

# Configuração dos argumentos de comando
parser = argparse.ArgumentParser(description='Treinamento do modelo U-Net para segmentação de imagens.')
parser.add_argument('--rgb', type=str, required=True, help='Caminho do diretório contendo as imagens RGB.')
parser.add_argument('--groundtruth', type=str, required=True, help='Caminho do diretório contendo as imagens segmentadas.')
parser.add_argument('--modelpath', type=str, required=True, help='Caminho para salvar o modelo treinado.')

# Análise dos argumentos fornecidos no terminal
args = parser.parse_args()

# Define o diretório padrão para salvar os gráficos das métricas
plot_directory = './train/metricas'

# Cria o diretório para salvar o gráfico, se não existir
os.makedirs(plot_directory, exist_ok=True)

# Carrega e pré-processa os dados
X_train, X_val, y_train, y_val = load_data(args.rgb, args.groundtruth)

# Define o modelo U-Net
input_shape = X_train.shape[1:]  # Obtém a forma da imagem
model = unet_model(input_shape)

# Compila o modelo especificando o otimizador, função de perda e acurácia
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Mensagem indicando o início do treinamento
print("Iniciando o treinamento do modelo...")

# Treina o modelo usando os dados de treinamento e validação
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=30, batch_size=8)

# Salva o modelo treinado
model.save(args.modelpath)

print(f"Modelo salvo em {args.modelpath}")

# Plota a evolução da perda
plt.figure(figsize=(12, 5))

# Subplot para perda
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Perda de Treinamento')
plt.plot(history.history['val_loss'], label='Perda de Validação')
plt.title('Evolução da Perda')
plt.xlabel('Épocas')
plt.ylabel('Perda')
plt.legend()

# Subplot para acurácia
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Acurácia de Treinamento')
plt.plot(history.history['val_accuracy'], label='Acurácia de Validação')
plt.title('Evolução da Acurácia')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()

# Salva o gráfico no diretório especificado
plot_file_path = os.path.join(plot_directory, 'metrica.png')
plt.savefig(plot_file_path)
plt.show()

print(f"Gráfico salvo em {plot_file_path}")
