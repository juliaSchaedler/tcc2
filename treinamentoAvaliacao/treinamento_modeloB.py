import numpy as np
import tensorflow as tf
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from astropy.io import fits
import cv2
import glob
import os
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix  # Importações adicionadas
import time

class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

# Configuração do logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuração da GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        logging.info("GPU habilitada para treinamento.")
    except RuntimeError as e:
        logging.error(f"Erro ao configurar a GPU: {e}")

# Funções auxiliares

def carregar_imagem_fits(caminho_fits):
    try:
        with fits.open(caminho_fits) as hdul:
            imagem = hdul[0].data
        if imagem is None:
            raise ValueError("Imagem vazia.")
        return imagem
    except Exception as e:
        logging.error(f"Erro ao carregar imagem FITS: {e}")
        return None
    
# Funções auxiliares (mantidas)

def preprocess_image(image, target_size, output_dir):
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    if image.shape[-1] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0  

    # Aumentar o Data Augmentation
    data_gen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,  # Novo
        brightness_range=[0.8, 1.2],  # Novo
        channel_shift_range=0.1,  # Novo
        fill_mode='nearest'
    )

    augmented_images = [data_gen.random_transform(image) for _ in range(5)]
    
    for i, aug_img in enumerate(augmented_images):
        augmented_path = os.path.join(output_dir, f'augmented_{i}.png')
        cv2.imwrite(augmented_path, aug_img * 255)

    return image

def carregar_rotulos_metadados_fits(caminho_rotulos):
    try:
        with fits.open(caminho_rotulos) as hdul:
            dados = hdul[1].data
        rotulos = np.stack((dados['SPIRAL'], dados['ELLIPTICAL'], dados['UNCERTAIN']), axis=-1)
        metadados = np.stack((dados['ra'], dados['dec'], dados['z'], dados['z_err'], dados['mag_auto_g'], dados['mag_auto_r'], dados['mag_auto_i'], dados['mass'], dados['mass_err'], dados['P_EL'], dados['P_CW'], dados['P_ACW'], dados['P_EDGE']), axis=-1)
        return rotulos, metadados
    except Exception as e:
        logging.error(f"Erro ao carregar rótulos ou metadados: {e}")
        return None, None

def carregar_metadados_antigos():
    caminho_fits_antigo = 'datasets_processados/dataset_preprocessado.fits'
    
    try:
        with fits.open(caminho_fits_antigo) as hdul:
            metadados_antigos = hdul[1].data
            colunas_disponiveis = ['pE', 'pS', 'MORPH_FLAG', 'MAG_AUTO_G', 'MAG_AUTO_R', 'MAG_AUTO_I', 'MAG_AUTO_Z', 'MAG_AUTO_Y', 'FLUX_RADIUS_I', 'FLUX_RADIUS_R']
            metadados = []
            for coluna in colunas_disponiveis:
                if coluna in metadados_antigos.columns.names:
                    metadados.append(metadados_antigos[coluna])
                else:
                    logging.warning(f"Coluna '{coluna}' não encontrada.")
                    metadados.append(None)
            
            metadados_antigos = np.stack([m for m in metadados if m is not None], axis=-1)
            return np.array(metadados_antigos)
        
    except Exception as e:
        logging.error(f"Erro ao carregar metadados do arquivo .fits: {e}")
        return None

def balancear_classes(X, y):
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X.reshape(X.shape[0], -1), y)
    X_resampled = X_resampled.reshape(-1, 64, 64, 3)  # Reformatando após o balanceamento
    return X_resampled, y_resampled

# Adicionar Early Stopping e reduzir a taxa de aprendizado
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

def criar_modelo(input_shape):
    base_model = tf.keras.applications.ResNet50(
        include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')
    
    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l1_l2(0.001)),  # L1 e L2
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),  # Aumento do Dropout

        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),  # Aumento do Dropout

        tf.keras.layers.Dense(2, activation='softmax', dtype='float32')
    ])
    
    return model

def plot_matriz_confusao(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusão')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Espiral", "Elíptica"], rotation=45)
    plt.yticks(tick_marks, ["Espiral", "Elíptica"])
    plt.ylabel('Rótulo verdadeiro')
    plt.xlabel('Rótulo previsto')
    plt.show()

def plot_graficos_avaliacao(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Perda Treino')
    plt.plot(history.history['val_loss'], label='Perda Validação')
    plt.title('Perda durante o Treinamento')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Acurácia Treino')
    plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
    plt.title('Acurácia durante o Treinamento')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()

    plt.show()

# Função principal
if __name__ == '__main__':
    caminho_saida_imagens = 'imagens_preprocessadas'
    os.makedirs(caminho_saida_imagens, exist_ok=True)
    
    caminhos_fits = glob.glob('img_DESY1_stripe82_GZ1/img_DESY1_stripe82_GZ1/*.fits')
    X, y = [], []

    for caminho_fits in tqdm(caminhos_fits, desc='Carregando e processando imagens'):
        imagem = carregar_imagem_fits(caminho_fits)
        if imagem is not None:
            imagem_preprocessada = preprocess_image(imagem, target_size=(64, 64), output_dir=caminho_saida_imagens)
            X.append(imagem_preprocessada)

    X = np.array(X)
    
    rotulos, metadados = carregar_rotulos_metadados_fits('desY1stripe82_GZ1_ES.fits')

    # Excluir amostras com rótulo "uncertain"
    rotulos_validos = (rotulos[:, 0] == 1) | (rotulos[:, 1] == 1)  # 1 para espiral ou elíptica
    X = X[rotulos_validos]
    rotulos = rotulos[rotulos_validos]

    # Convertendo rótulos para classes binárias
    y = np.where(rotulos[:, 0] == 1, 0, 1)  # 0 para espiral, 1 para elíptica

    # Balanceamento de classes
    print(f"Dimensão de X antes do balanceamento: {X.shape}")
    print(f"Dimensão de y antes do balanceamento: {y.shape}")
    X_resampled, y_resampled = balancear_classes(X, y)

    # Divisão em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    input_shape = X_train.shape[1:]
    model = criar_modelo(input_shape)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=100, batch_size=32,
                        callbacks=[early_stopping, reduce_lr, ModelCheckpoint('modeloB.h5', save_best_only=True)])
    
    # Avaliação final
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    plot_matriz_confusao(y_test, y_pred_classes)
    plot_graficos_avaliacao(history)
    print(classification_report(y_test, y_pred_classes))
