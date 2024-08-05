'''
Descrição: O script define a arquitetura baseada em rede neural U-Net para segmentação 
           de imagens, composta por blocos de contração para extração de características 
           e blocos de expansão para reconstrução da imagem. O modelo é configurado para 
           produzir uma máscara binária.
Data:      05/08/2024
'''

# Importação da biblioteca tensorflow para construção do modelo
import tensorflow as tf

# Função da arquiteura do modelo
def unet_model(input_shape):
    
    # Entrada do modelo
    inputs = tf.keras.Input(shape=input_shape)
    
    # Camadas de convolução e pooling para extrair características
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    
    # Camadas de convolução transposta e concatenação para reconstruir a imagem
    up5 = tf.keras.layers.Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv4)
    up5 = tf.keras.layers.concatenate([up5, conv3])
    conv5 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(up5)
    conv5 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv5)
    
    up6 = tf.keras.layers.Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv5)
    up6 = tf.keras.layers.concatenate([up6, conv2])
    conv6 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(up6)
    conv6 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv6)
    
    up7 = tf.keras.layers.Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv6)
    up7 = tf.keras.layers.concatenate([up7, conv1])
    conv7 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(up7)
    conv7 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv7)
    
    # Camada de saída com uma convolução 1x1 para a classificação de cada pixel
    outputs = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv7)
    
    # Modelo com as entradas e saídas especificadas
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    return model
