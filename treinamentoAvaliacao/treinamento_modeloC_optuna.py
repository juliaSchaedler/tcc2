import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
import optuna
from optuna.integration import KerasPruningCallback
from optuna.trial import TrialState
from sklearn.ensemble import RandomForestRegressor
from optuna.samplers import TPESampler
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from astropy.io import fits
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from scipy.spatial import cKDTree
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from sklearn.ensemble import RandomForestRegressor
    
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        logging.info("GPU habilitada para treinamento.")
    except RuntimeError as e:
        logging.error(f"Erro ao configurar a GPU: {e}")

# Data loading and preprocessing functions
def carregar_imagem_fits(caminho_fits):
    try:
        with fits.open(caminho_fits) as hdul:
            imagem = hdul[0].data
        if imagem is None:
            raise ValueError("Imagem vazia.")
        return imagem
    except Exception as e:
        logger.error(f"Erro ao carregar imagem FITS: {e}")
        return None

def preprocess_image(image, target_size, output_dir):
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    if image.shape[-1] != 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(image, target_size)

    # Ajustes no pré-processamento
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.medianBlur(image_gray, 3)
    image_gray = cv2.Canny(image_gray, 50, 150)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_clahe = clahe.apply(image_gray)
    image_norm = cv2.normalize(image_clahe, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    image_norm = (image_norm - np.mean(image_norm)) / np.std(image_norm)
    image_norm = cv2.normalize(image_norm, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    image_processed = cv2.cvtColor(image_norm, cv2.COLOR_GRAY2BGR)

    # Data augmentation
    data_gen = ImageDataGenerator(
        rotation_range=60,
        width_shift_range=0.5,
        height_shift_range=0.5,
        shear_range=0.4,
        zoom_range=0.5,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.5, 1.5],
        channel_shift_range=0.4,
        fill_mode='nearest'
    )

    augmented_images = [data_gen.random_transform(image_processed) for _ in range(35)]

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
        logger.error(f"Erro ao carregar rótulos ou metadados: {e}")
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
                    logger.warning(f"Coluna '{coluna}' não encontrada.")
                    metadados.append(None)
            metadados_antigos = np.stack([m for m in metadados if m is not None], axis=-1)
            return np.array(metadados_antigos)
    except Exception as e:
        logger.error(f"Erro ao carregar metadados do arquivo .fits: {e}")
        return None

def balancear_classes(X, y):
    # Combinação de SMOTE e undersampling
    over = SMOTE(sampling_strategy=0.8, random_state=42)
    under = RandomUnderSampler(sampling_strategy=1.0, random_state=42)
    X_resampled, y_resampled = over.fit_resample(X.reshape(X.shape[0], -1), y)
    X_resampled, y_resampled = under.fit_resample(X_resampled, y_resampled)
    X_resampled = X_resampled.reshape(-1, 128, 128, 3)
    return X_resampled, y_resampled

def carregar_e_processar_imagem(args):
    caminho_fits, caminho_saida_imagens = args
    imagem = carregar_imagem_fits(caminho_fits)
    if imagem is not None:
        imagem_preprocessada = preprocess_image(imagem, target_size=(128, 128), output_dir=caminho_saida_imagens)
        return imagem_preprocessada
    return None

def distancia_angular(ra1, dec1, ra2, dec2):
    ra1_rad = np.radians(ra1, dtype=np.float64)
    dec1_rad = np.radians(dec1, dtype=np.float64)
    ra2_rad = np.radians(ra2, dtype=np.float64)
    dec2_rad = np.radians(dec2, dtype=np.float64)

    arg_arccos = np.sin(dec1_rad) * np.sin(dec2_rad) + np.cos(dec1_rad) * np.cos(dec2_rad) * np.cos(ra1_rad - ra2_rad)
    arg_arccos = np.clip(arg_arccos, -1, 1)

    return np.degrees(np.arccos(arg_arccos))

def calcular_distancias_batch(ra1, dec1, metadados_antigos_escalados, limite_distancia=0.001, batch_size=10000):
    ra1_rad = np.radians(ra1)
    dec1_rad = np.radians(dec1)

    kd_tree = cKDTree(np.column_stack((np.radians(metadados_antigos_escalados[:, 7]), np.radians(metadados_antigos_escalados[:, 8]))))
    _, indices = kd_tree.query(np.column_stack((ra1_rad, dec1_rad)), k=1)

    distances = distancia_angular(ra1, dec1, metadados_antigos_escalados[indices, 7], metadados_antigos_escalados[indices, 8])
    if np.all(distances <= limite_distancia):
        return indices
    else:
        return None

def calcular_distancias_unpack(args):
    return calcular_distancias_batch(*args)

def criar_modelo(input_shape, num_metadados_imagem, num_metadados_antigos, learning_rate, dropout_rate=0.5, l2_reg=0.01):
    input_imagem = tf.keras.Input(shape=input_shape)

    def bloco_residual(x, filtros, kernel_size=(3, 3)):
        y = tf.keras.layers.Conv2D(filtros, kernel_size, activation='elu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.Conv2D(filtros, kernel_size, activation=None, padding='same', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(y)
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.Add()([x, y])
        y = tf.keras.layers.Activation('relu')(y)
        return y

    x = tf.keras.layers.Conv2D(32, (3, 3), activation='elu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(input_imagem)
    x = tf.keras.layers.BatchNormalization()(x)
    x = bloco_residual(x, 32)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Conv2D(64, (3, 3), activation='elu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = bloco_residual(x, 64)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = bloco_residual(x, 128)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = bloco_residual(x, 256)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    x = tf.keras.layers.Flatten()(x)

    input_metadados_imagem = tf.keras.Input(shape=(num_metadados_imagem,))
    input_metadados_antigos = tf.keras.Input(shape=(num_metadados_antigos,))

    x = tf.keras.layers.concatenate([x, input_metadados_imagem, input_metadados_antigos])

    x = tf.keras.layers.Dense(1024, activation=tf.keras.layers.LeakyReLU(), kernel_regularizer=tf.keras.regularizers.l2(0.03))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.7)(x)

    output = tf.keras.layers.Dense(2, activation='softmax', dtype='float32')(x)


    # --- Opções de otimizadores ---
    initial_learning_rate = learning_rate  
    
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=10000,
        decay_rate=0.96,
        staircase=True)

    #optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9, nesterov=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # optimizer = AdamW(learning_rate=lr_schedule, weight_decay=1e-4)  # AdamW com weight decay

    model = tf.keras.Model(inputs=[input_imagem, input_metadados_imagem, input_metadados_antigos], outputs=output)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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
    
    
# Função para otimizar os hiperparâmetros do modelo usando Optuna
def otimizar_hiperparametros_optuna(X_train, y_train, metadados_imagem_train, metadados_antigos_train, X_test, metadados_imagem_test, metadados_antigos_test, y_test):
    def objective(trial):
        # Hiperparâmetros a serem otimizados
        learning_rate = trial.suggest_float('learning_rate', 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_int('batch_size', 32, 128)
        epochs = trial.suggest_int('epochs', 50, 500)
        l2_reg = trial.suggest_float('l2_reg', 0.01, 0.05, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
        
        

        # Criar o modelo
        modelo = criar_modelo(
            (128, 128, 3),
            metadados_imagem_train.shape[1],
            metadados_antigos_train.shape[1], 
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            l2_reg=l2_reg
        )

        # Callbacks para Early Stopping e Pruning
        early_stopping = EarlyStopping(monitor='val_loss', patience=5)
        pruning_callback = KerasPruningCallback(trial, monitor='val_loss')

        # Treinar o modelo
        history = modelo.fit(
            [X_train, metadados_imagem_train, metadados_antigos_train],
            y_train,
            validation_data=(
                [X_test, metadados_imagem_test, metadados_antigos_test],
                y_test
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, pruning_callback]
        )

        # Retornar a perda na validação
        return history.history['val_loss'][-1]

    # Criar um estudo Optuna
    study = optuna.create_study(
        direction='minimize',
        sampler=optuna.samplers.TPESampler(multivariate=True, seed=42),
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1,
            reduction_factor=4,
            min_early_stopping_rate=0
        )
    )

    # Otimizar o estudo
    study.optimize(objective, n_trials=10, n_jobs=-1)

    # Imprimir os resultados
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Estudo otimizado:")
    print("  Número de tentativas finalizadas:", len(study.trials))
    print("  Número de tentativas podadas:", len(pruned_trials))
    print("  Número de tentativas completas:", len(complete_trials))

    print("Melhor tentativa:")
    trial = study.best_trial

    print("  Valor:", trial.value)
    print("  Parâmetros: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return study.best_params

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Novo diretório para imagens pré-processadas
    caminho_saida_imagens = 'img_pre_processadas_2'
    os.makedirs(caminho_saida_imagens, exist_ok=True)

    caminhos_fits = glob.glob('img_DESY1_stripe82_GZ1/img_DESY1_stripe82_GZ1/*.fits')

    # --- Carregar e processar as imagens em paralelo ---
    with ThreadPoolExecutor() as executor:
        X = list(executor.map(carregar_e_processar_imagem, [(caminho_fits, caminho_saida_imagens) for caminho_fits in caminhos_fits]))

    X = np.array([x for x in X if x is not None]) 

    rotulos, metadados_imagem = carregar_rotulos_metadados_fits('desY1stripe82_GZ1_ES.fits')

    # Excluir amostras com rótulo "uncertain"
    rotulos_validos = (rotulos[:, 0] == 1) | (rotulos[:, 1] == 1)
    X = X[rotulos_validos]
    rotulos = rotulos[rotulos_validos]
    metadados_imagem = metadados_imagem[rotulos_validos]

    # Convertendo rótulos para classes binárias
    y = np.where(rotulos[:, 0] == 1, 0, 1)

    # --- Carregar e pré-processar os metadados antigos ---
    metadados_antigos = carregar_metadados_antigos()

    # --- Pré-processar os metadados da imagem ---
    scaler = StandardScaler()
    metadados_imagem_escalados = scaler.fit_transform(metadados_imagem)

    # --- Pré-processar os metadados antigos ---
    scaler = StandardScaler()
    metadados_antigos_escalados = scaler.fit_transform(metadados_antigos)

    # --- Definir um limite de distância para considerar as galáxias como correspondentes (em graus) ---
    limite_distancia = 0.001  
    # --- Calcular as distâncias em paralelo, em batches ---
    with ThreadPoolExecutor() as executor:
        indices_correspondentes = list(executor.map(calcular_distancias_unpack,
                                                    zip(metadados_imagem_escalados[:, 0], 
                                                        metadados_imagem_escalados[:, 1], 
                                                        [metadados_antigos_escalados] * len(metadados_imagem_escalados))))

    # --- Filtrar a lista indices_correspondentes para remover os None ---
    indices_correspondentes = [indice for indice in indices_correspondentes if indice is not None]

    # Filtrar os metadados antigos para conter apenas as correspondências encontradas
    metadados_antigos_filtrados = metadados_antigos[indices_correspondentes]

    # --- Pré-processar os metadados antigos ---
    scaler = StandardScaler()
  
    print(f"Formato original: {metadados_antigos_filtrados.shape}")
    
    metadados_antigos_filtrados = metadados_antigos_filtrados.reshape(metadados_antigos_filtrados.shape[0], -1)   
    metadados_antigos_escalados = scaler.fit_transform(metadados_antigos_filtrados)

    # Balanceamento de classes (SMOTE + undersampling)
    print(f"Dimensão de X antes do balanceamento: {X.shape}")
    print(f"Dimensão de y antes do balanceamento: {y.shape}")
    X_resampled, y_resampled = balancear_classes(X, y)
    print(f"Dimensão de X depois do balanceamento: {X_resampled.shape}")
    print(f"Dimensão de y depois do balanceamento: {y_resampled.shape}")

    # --- Ajustar os metadados para o mesmo tamanho de X_resampled e y_resampled ---
    metadados_imagem_resampled = resample(metadados_imagem_escalados, n_samples=len(X_resampled), random_state=42)
    metadados_antigos_resampled = resample(metadados_antigos_escalados, n_samples=len(X_resampled), random_state=42)

    # Divisão em treino e teste
    X_train, X_test, \
    y_train, y_test, \
    metadados_imagem_train, metadados_imagem_test, \
    metadados_antigos_train, metadados_antigos_test = train_test_split(
        X_resampled, y_resampled,
        metadados_imagem_resampled,
        metadados_antigos_resampled,
        test_size=0.2, random_state=42
    )

    # Chamar a função de otimização com Optuna
    melhores_parametros = otimizar_hiperparametros_optuna(
        X_train, y_train, metadados_imagem_train, metadados_antigos_train, 
        X_test, metadados_imagem_test, metadados_antigos_test, y_test
    )

    # --- Criar o modelo com os melhores hiperparâmetros ---
    batch_size = melhores_parametros['batch_size']  
    epochs = melhores_parametros['epochs']  

    melhores_parametros.pop('batch_size')  
    melhores_parametros.pop('epochs')  

    
    modelo = criar_modelo((128, 128, 3), metadados_imagem_train.shape[1], metadados_antigos_train.shape[1], 
                         **melhores_parametros)  

    # --- Treinar o modelo final ---

    checkpoint = ModelCheckpoint('modelo_C_Optuna.keras', monitor='val_loss', save_best_only=True, mode='min')
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

    logger.info('Iniciando o treinamento do modelo final...')

    history = modelo.fit(
        [X_train, metadados_imagem_train, metadados_antigos_train], y_train,
        validation_data=([X_test, metadados_imagem_test, metadados_antigos_test], y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, reduce_lr, checkpoint]
    )

    logger.info('Treinamento do modelo finalizado.')


    # Avaliação
    y_pred = np.argmax(modelo.predict([X_test, metadados_imagem_test, metadados_antigos_test]), axis=-1)  
    plot_matriz_confusao(y_test, y_pred)
    plot_graficos_avaliacao(history)

    print(classification_report(y_test, y_pred, target_names=["Espiral", "Elíptica"]))

    # --- Avaliação do pré-processamento ---
   
    def visualizar_imagens(X_original, X_preprocessado, num_imagens=5):
        plt.figure(figsize=(12, 4))
        for i in range(num_imagens):
            plt.subplot(2, num_imagens, i + 1)
            plt.imshow(X_original[i])
            plt.title('Original')
            plt.axis('off')

            plt.subplot(2, num_imagens, i + 1 + num_imagens)
            plt.imshow(X_preprocessado[i])
            plt.title('Pré-processada')
            plt.axis('off')
        plt.show()

    
    visualizar_imagens(X, X_resampled)

    
    def plotar_histograma_pixels(imagens):
        plt.figure(figsize=(8, 6))
        plt.hist(imagens.ravel(), bins=256, range=(0, 1), density=True)
        plt.xlabel('Valor do pixel')
        plt.ylabel('Frequência')
        plt.title('Histograma dos valores dos pixels')
        plt.show()

    
    plotar_histograma_pixels(X_resampled)

if __name__ == '__main__':
    main()
