'''
Descrição: O script divide uma imagem TIFF em blocos (frames) menores do tamanho especificado 
           e salva cada bloco como um arquivo PNG. Cria um diretório de saída e processa a 
           imagem em blocos, salvando cada um com um nome sequencial.
Data:      05/08/2024
'''

# Importação das bibliotecas
import os  # Manipulação de arquivos e diretórios
import argparse  # Argumentos na linha de comando
import cv2  # Manipulação de dados de imagem

# Configuração dos argumentos de comando
parser = argparse.ArgumentParser(description='Divisão de imagem TIFF em blocos.')
parser.add_argument('--input', type=str, required=True, help='Caminho de entrada da imagem TIFF.')
parser.add_argument('--output', type=str, required=True, help='Diretório de saída das imagens.')
args = parser.parse_args()

# Tamanho do bloco
frame_size = (256, 256)  # (largura, altura)

# Carrega a imagem TIFF
image = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
image_height, image_width = image.shape[:2]

# Cria o diretório de saída, se não existir
os.makedirs(args.output, exist_ok=True)

# Obtém o tamanho do bloco
frame_width, frame_height = frame_size

# Contador para os blocos
count = 0

# Divide a imagem em blocos e salva cada um como um arquivo PNG
for x1 in range(0, image_width, frame_width):  # Itera sobre a altura da imagem
    for y1 in range(0, image_height, frame_height):  # Itera sobre a largura da imagem
        # Define a caixa de recorte para o bloco
        x2 = min(x1 + frame_width, image_width)  # Cálculo da coordenada x final
        y2 = min(y1 + frame_height, image_height)  # Cálculo da coordenada y final

        # Recorta o bloco da imagem
        frame = image[y1:y2, x1:x2]

        # Verifica se o bloco tem o tamanho correto
        if frame.shape[0] == frame_height and frame.shape[1] == frame_width:
            # Define o nome do arquivo e salva como PNG
            frame_filename = os.path.join(args.output, f"frame_{count:04d}.png")
            cv2.imwrite(frame_filename, frame)

            # Exibe uma mensagem de progresso
            print(f"Frame {count:04d} salvo como {frame_filename}")

            # Contador de blocos processados
            count += 1
        
# Exibe o número total de blocos salvos
print(f"Total de imagens salvas: {count}")

