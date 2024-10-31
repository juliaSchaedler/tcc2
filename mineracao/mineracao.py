import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from astropy.io import fits
from tensorflow.keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os

# Função para carregar rótulos reais do arquivo FITS
def carregar_rotulos(arquivo_fits):
    with fits.open(arquivo_fits) as hdul:
        dados = hdul[1].data
        rotulos_espiral = dados['SPIRAL']
        rotulos_eliptica = dados['ELLIPTICAL']
        rotulos_reais = np.where(rotulos_espiral == 1, 1, np.where(rotulos_eliptica == 1, 0, -1))
    return rotulos_reais

# Função para carregar as imagens e metadados do catálogo DES
def carregar_dados(diretorio_imagens, arquivo_metadados):
    imagens, arquivos = [], sorted(os.listdir(diretorio_imagens))
    for arquivo in arquivos:
        if arquivo.endswith('.fits'):
            with fits.open(os.path.join(diretorio_imagens, arquivo)) as hdul:
                imagens.append(hdul[0].data)
    with fits.open(arquivo_metadados) as hdul:
        metadados = pd.DataFrame(np.array(hdul[1].data).byteswap().newbyteorder())
    return np.array(imagens), metadados

# Função para carregar o modelo treinado
def carregar_modelo(caminho_modelo):
    return load_model(caminho_modelo)

# Função para classificar galáxias usando CNN
def classificar_galaxias(imagens, modelo):
    imagens_expandidas = np.expand_dims(imagens, axis=-1)
    predicoes = modelo.predict(imagens_expandidas)
    return np.argmax(predicoes, axis=1)

# Função para avaliar a performance do modelo
def avaliar_performance(rotulos_reais, predicoes):
    print("Matriz de Confusão:")
    print(confusion_matrix(rotulos_reais, predicoes))
    print("\nRelatório de Classificação:")
    print(classification_report(rotulos_reais, predicoes, target_names=['Elíptica', 'Espiral']))
    accuracy = accuracy_score(rotulos_reais, predicoes)
    precision = precision_score(rotulos_reais, predicoes, average='macro')
    recall = recall_score(rotulos_reais, predicoes, average='macro')
    f1 = f1_score(rotulos_reais, predicoes, average='macro')
    print(f'Acurácia: {accuracy:.2f}, Precisão: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}')

# Função para explorar metadados com visualizações
def explorar_dados_interativos(metadados):
    variaveis_interesse = ['MAG_AUTO_G', 'MAG_AUTO_R', 'Z_PHOTO']
    for variavel in variaveis_interesse:
        plt.figure()
        metadados[variavel].hist(bins=50)
        plt.title(f"Distribuição de {variavel}")
        plt.xlabel(variavel)
        plt.ylabel('Frequência')
        plt.show()
    fig = px.scatter_matrix(metadados[variaveis_interesse], title="Análise Interativa entre Variáveis")
    fig.show()

# Função para análise de clusterização
def clusterizar_imagens(imagens, metodo='DBSCAN', n_clusters=5, eps=0.5, min_samples=5):
    pca = PCA(n_components=50)
    imagens_pca = pca.fit_transform(imagens.reshape(imagens.shape[0], -1))
    if metodo == 'DBSCAN':
        clusters = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(imagens_pca)
    else:
        clusters = KMeans(n_clusters=n_clusters).fit_predict(imagens_pca)
    return clusters

# Função para detectar anomalias com Isolation Forest
def detectar_anomalias(imagens, contaminação=0.05):
    iforest = IsolationForest(contamination=contaminação)
    anomalias = iforest.fit_predict(imagens.reshape(imagens.shape[0], -1))
    return anomalias

# Função para previsão do brilho usando suavização exponencial
def prever_brilho(metadados):
    modelo = ExponentialSmoothing(metadados['MAG_AUTO_G'], trend="add", seasonal="add", seasonal_periods=12)
    ajuste = modelo.fit()
    previsao = ajuste.forecast(12)
    plt.plot(metadados['MAG_AUTO_G'], label="Dados Reais")
    plt.plot(range(len(metadados['MAG_AUTO_G']), len(metadados['MAG_AUTO_G']) + 12), previsao, color='red', label="Previsão")
    plt.title("Previsão de Brilho")
    plt.legend()
    plt.show()

# Função para exibir clusters e anomalias detectadas
def exibir_resultados(imagens, clusters, titulo="Resultados"):
    unicos = np.unique(clusters)
    plt.figure(figsize=(15, 15))
    for i, cluster in enumerate(unicos):
        plt.subplot(5, 5, i + 1)
        indices = np.where(clusters == cluster)[0]
        plt.imshow(imagens[indices[0]], cmap='gray')
        plt.title(f"{titulo} {cluster}")
        plt.axis('off')
    plt.suptitle(titulo, fontsize=16)
    plt.show()

# Pipeline completo de análise
def executar_pipeline(diretorio_imagens, arquivo_rotulos, caminho_modelo_cnn, arquivo_metadados):
    # Carregar dados e modelo
    rotulos_reais = carregar_rotulos(arquivo_rotulos)
    imagens, metadados = carregar_dados(diretorio_imagens, arquivo_metadados)
    modelo = carregar_modelo(caminho_modelo_cnn)

    # Classificação das galáxias
    predicoes = classificar_galaxias(imagens, modelo)
    avaliar_performance(rotulos_reais, predicoes)

    # Análise interativa dos metadados
    explorar_dados_interativos(metadados)

    # Clusterização e detecção de anomalias
    clusters_dbscan = clusterizar_imagens(imagens, metodo='DBSCAN')
    exibir_resultados(imagens, clusters_dbscan, titulo="Clusters DBSCAN")
    clusters_kmeans = clusterizar_imagens(imagens, metodo='KMeans', n_clusters=5)
    exibir_resultados(imagens, clusters_kmeans, titulo="Clusters KMeans")
    anomalias = detectar_anomalias(imagens)
    exibir_resultados(imagens, anomalias, titulo="Anomalias")

    # Previsão de brilho
    prever_brilho(metadados)

# Caminhos de arquivos
diretorio_imagens = 'img_DESY1_stripe82_GZ1/img_DESY1_stripe82_GZ1'  # Substitua com o caminho correto
arquivo_rotulos = 'desY1stripe82_GZ1_ES.fits'  # Substitua com o caminho correto
caminho_modelo_cnn = 'modelo_galaxia.h5'  # Substitua com o caminho correto
arquivo_metadados = 'datasets_processados/dataset_preprocessado.fits'

# Executar o pipeline
executar_pipeline(diretorio_imagens, arquivo_rotulos, caminho_modelo_cnn, arquivo_metadados)
