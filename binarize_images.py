'''
Descrição: O script binariza os frames utilizando o índice ExG (Excess Green Index), 
           para detectar vegetação. 
           Configura argumentos para entrada e saída de diretórios, processa as imagens 
           conforme o índice ExG, e salva os resultados binarizados em um diretório de saída.
Data:      05/08/2024
'''

# Importação das bibliotecas
import os  # Manipulação de arquivos e diretórios
import argparse  # Argumentos na linha de comando
import cv2  # Manipulação de dados de imagem
import numpy as np  # Operações numéricas

# Configuração dos argumentos de comando
parser = argparse.ArgumentParser(description='Binarização de imagens de blocos para detecção de vegetação.')
parser.add_argument('--input', type=str, required=True, help='Caminho do diretório contendo as imagens RGB.')
parser.add_argument('--output', type=str, required=True, help='Diretório de saída para imagens binarizadas.')

# Análise dos argumentos fornecidos no terminal
args = parser.parse_args()

# Função que calcula o índice ExG (Excess Green Index) para cada pixel da imagem RGB.
# Formula: ExG = 2 * G - R - B
def calculate_exg(image):
    
    # Converte a imagem para o formato RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Extrai os canais de cor R (vermelho), G (verde) e B (azul)
    R, G, B = image_rgb[:, :, 0], image_rgb[:, :, 1], image_rgb[:, :, 2]
    
    # Calcula o índice ExG
    exg = 2 * G - R - B
    
    return exg

# Função que binaliza a imagem
# Pixels com ExG acima do limiar são definidos como 1 (vegetação),
# e pixels com ExG abaixo do limiar são definidos como 0 (sem vegetação).
def binarize_image(image, threshold=30):
   
    # Calcula o índice ExG
    exg = calculate_exg(image)
    
    # Binariza com base no limiar
    _, binary_image = cv2.threshold(exg, threshold, 1, cv2.THRESH_BINARY)
    
    # Converte a imagem binarizada para o formato uint8 (0 e 1)
    binary_image = binary_image.astype(np.uint8)
    
    return binary_image

# Função que processa e binariza as imagens no diretório de entrada e salva no diretório de saída.
def process_images(input_dir, output_dir):
   
    # Cria o diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Itera sobre todos os arquivos no diretório de entrada
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg')):
            # Constrói o caminho completo do arquivo
            file_path = os.path.join(input_dir, filename)
            
            # Abre a imagem
            image = cv2.imread(file_path)
            
            # Verifica se a imagem foi carregada corretamente
            if image is not None:
                # Binariza a imagem
                binary_image = binarize_image(image)
                
                # Constrói o caminho completo do arquivo de saída
                output_path = os.path.join(output_dir, filename)
                
                # Salva a imagem binarizada como PNG
                cv2.imwrite(output_path, binary_image * 255)  # Multiplica por 255 para salvar corretamente como imagem de 8 bits
                
                # Exibe uma mensagem de progresso
                print(f"Imagem binarizada salva como {output_path}")
            else:
                print(f"Não foi possível carregar a imagem {file_path}")

    print("Processamento concluído.")

# Chama a função principal com os argumentos fornecidos
process_images(args.input, args.output)
