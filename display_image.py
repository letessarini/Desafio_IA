'''
Descrição: O script exibe uma imagem TIFF e seu tamanho
Data:      05/08/2024
'''

# Importação das bibliotecas
import cv2  # Manipulação de imagens
import matplotlib.pyplot as plt  # Visualização de imagens
import argparse  # Aargumentos de linha de comando

# Configuração dos argumentos que o script aceitará na linha de comando
parser = argparse.ArgumentParser(description='Exibição de imagem TIFF e suas informações básicas.')
parser.add_argument('--image_path', type=str, help='Caminho para a imagem TIFF')

# Análise dos argumentos fornecidos pelo usuário no terminal
args = parser.parse_args()

# Caminho para a imagem é obtido dos argumentos fornecidos
image_path = args.image_path

# Carrega a imagem TIFF usando OpenCV
# cv2.IMREAD_UNCHANGED carrega a imagem com o formato original sem alterações
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# OpenCV carrega imagens em formato BGR por padrão, e converte para RGB para visualização com matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Exibe a imagem usando matplotlib
plt.imshow(image_rgb)  # Exibe a imagem RGB
plt.axis('off')  # Remove os eixos da visualização
plt.show()  # Mostra a imagem

# Obtém as dimensões da imagem (altura e largura)
height, width = image.shape[:2]

# Imprime informações básicas da imagem
# Tamanho: Largura x Altura
print(f"Tamanho: {width}x{height}")
