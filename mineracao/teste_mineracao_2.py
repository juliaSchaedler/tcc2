import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Certifique-se de que o backend está configurado
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import IsolationForest
import cv2
import os
from astropy.io import fits
import seaborn as sns
import plotly.express as px


def carregar_rotulos(fits_file):
    with fits.open(fits_file) as hdul:
        data = hdul[1].data
        print("Nomes das colunas no arquivo FITS:", data.columns)

        rotulos = {}
        for i in range(len(data)):
            nome_imagem = data[i]['filename']
            if data[i]['SPIRAL'] == 1:
                rotulos[nome_imagem] = 'SPIRAL'
            elif data[i]['ELLIPTICAL'] == 1:
                rotulos[nome_imagem] = 'ELLIPTICAL'
            else:
                rotulos[nome_imagem] = 'UNCERTAIN'
    return rotulos


def carregar_modelo(caminho_modelo):
    try:
        if caminho_modelo.endswith('.h5') or caminho_modelo.endswith('.keras'):
            modelo = load_model(caminho_modelo, compile=False)
            print("Modelo CNN carregado com sucesso.")
        elif caminho_modelo.endswith('.resnet50'):
            modelo = ResNet50(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
            print("Modelo ResNet50 carregado com sucesso.")
        else:
            raise ValueError("Formato de modelo não suportado.")
        return modelo
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return None


def carregar_imagem(caminho_imagem, colorido=False):
    if caminho_imagem.endswith('.fits'):
        with fits.open(caminho_imagem) as hdul:
            img_data = hdul[0].data
            if img_data is None:
                return None
            img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data)) * 255
            img_data = img_data.astype(np.uint8)
            if colorido:
                img_data = np.stack([img_data] * 3, axis=-1)
            return img_data
    else:
        img = plt.imread(caminho_imagem)
        return img


def prever_classes(modelo, diretorio_imagens, rotulos_verdadeiros):
    imagens = []
    predicoes = []
    nomes_imagens = []

    for imagem_nome in os.listdir(diretorio_imagens):
        caminho_imagem = os.path.join(diretorio_imagens, imagem_nome)
        img = carregar_imagem(caminho_imagem)

        if img is not None:
            img_resized = cv2.resize(img, (128, 128))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
            img_rgb = img_rgb.astype(np.float32) / 255.0

            imagens.append(img_rgb)
            pred = modelo.predict(np.expand_dims(img_rgb, axis=0))
            predicoes.append(np.argmax(pred, axis=1)[0])
            nomes_imagens.append(imagem_nome)

    fig, axes = plt.subplots(nrows=min(len(imagens), 10), ncols=1, figsize=(10, 20))
    for ax, img, pred, nome_imagem in zip(axes, imagens, predicoes, nomes_imagens):
        ax.imshow(img)
        rotulo_verdadeiro = rotulos_verdadeiros.get(nome_imagem, "Desconhecido")
        ax.set_title(f'Predição: {pred}, Rótulo Verdadeiro: {rotulo_verdadeiro}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    return imagens, predicoes


def visualizar_imagens_coloridas(diretorio_imagens):
    imagens_coloridas = []
    for nome_imagem in os.listdir(diretorio_imagens):
        caminho_imagem = os.path.join(diretorio_imagens, nome_imagem)
        img = carregar_imagem(caminho_imagem, colorido=True)
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        imagens_coloridas.append(img)
    
    plt.figure(figsize=(10, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(imagens_coloridas[i])
        plt.axis("off")
    plt.suptitle("Imagens Coloridas com Ajuste de Contraste")
    plt.show()


def prever_brilho(brilhos):
    modelo_brilho = ExponentialSmoothing(brilhos, trend="add", seasonal=None)
    ajuste_brilho = modelo_brilho.fit()
    previsoes_brilho = ajuste_brilho.forecast(10)
    return previsoes_brilho


def detectar_anomalias(dados):
    modelo_anomalias = IsolationForest(contamination=0.1, random_state=42)
    dados = np.array(dados).reshape(-1, 1)
    predicoes_anomalias = modelo_anomalias.fit_predict(dados)
    return predicoes_anomalias


def analisar_clusters(df_features):
    pca = PCA(n_components=2)
    dados_pca = pca.fit_transform(df_features)
    
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters_kmeans = kmeans.fit_predict(dados_pca)
    
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters_dbscan = dbscan.fit_predict(dados_pca)
    
    fig = px.scatter(x=dados_pca[:, 0], y=dados_pca[:, 1], color=clusters_kmeans.astype(str),
                     title="Clusters de Galáxias (KMeans)", labels={"x": "PCA 1", "y": "PCA 2"})
    fig.show()
    
    fig = px.scatter(x=dados_pca[:, 0], y=dados_pca[:, 1], color=clusters_dbscan.astype(str),
                     title="Clusters de Galáxias (DBSCAN)", labels={"x": "PCA 1", "y": "PCA 2"})
    fig.show()


def exibir_relatorio_classificacao(rotulos_verdadeiros, rotulos_previstos):
    from sklearn.metrics import classification_report
    print("Relatório de Classificação:\n")
    print(classification_report(rotulos_verdadeiros, rotulos_previstos, target_names=['Elíptica', 'Espiral']))


def executar_pipeline(arquivo_fits, diretorio_imagens, caminho_modelo_cnn, arquivo_metadados):
    modelo = carregar_modelo(caminho_modelo_cnn)
    if not modelo:
        return

    rotulos_reais = carregar_rotulos(arquivo_fits)

    imagens, predicoes = prever_classes(modelo, diretorio_imagens, rotulos_reais)
    
    visualizar_imagens_coloridas(diretorio_imagens)

    df_metadados = pd.read_csv(arquivo_metadados)
    brilhos = df_metadados['brilho'].values
    previsoes_brilho = prever_brilho(brilhos)
    plt.figure()
    plt.plot(brilhos, label='Brilho Observado')
    plt.plot(range(len(brilhos), len(brilhos) + len(previsoes_brilho)), previsoes_brilho, label='Brilho Previsto', linestyle='--')
    plt.legend()
    plt.title("Previsão do Brilho")
    plt.xlabel("Tempo")
    plt.ylabel("Brilho")
    plt.show()

    anomalias = detectar_anomalias(df_metadados['brilho'].values)
    plt.figure()
    plt.plot(brilhos, label='Brilho')
    plt.scatter(df_metadados.index, brilhos, c=anomalias, cmap='coolwarm', label='Anomalias')
    plt.legend()
    plt.title("Detecção de Anomalias no Brilho")
    plt.xlabel("Índice")
    plt.ylabel("Brilho")
    plt.show()

    analisar_clusters(df_metadados[['brilho', 'tamanho']])

    exibir_relatorio_classificacao(rotulos_reais, predicoes)


# Caminhos dos arquivos (atualize conforme necessário)
arquivo_fits = 'desY1stripe82_GZ1_ES.fits'
diretorio_imagens = 'img_DESY1_stripe82_GZ1/img_DESY1_stripe82_GZ1'
caminho_modelo_cnn = 'modeloA.h5'
arquivo_metadados = 'datasets_processados/dataset_preprocessado.fits'

# Executar o pipeline
executar_pipeline(arquivo_fits, diretorio_imagens, caminho_modelo_cnn, arquivo_metadados)
