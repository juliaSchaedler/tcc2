import os

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from tensorflow.keras.models import load_model


matplotlib.use('Qt5Agg')


def carregar_rotulos(fits_file):
    """Carrega os rótulos das galáxias a partir de um arquivo FITS."""
    with fits.open(fits_file) as hdul:
        data = hdul[1].data

        
        rotulos = {}

        for i in range(len(data)):
            nome_imagem = data[i]['filename']  
           
            if data[i]['SPIRAL'] == 1:
                rotulos[nome_imagem] = 'ESPIRAL'
            elif data[i]['ELLIPTICAL'] == 1:
                rotulos[nome_imagem] = 'ELÍPTICA'
            else:
                rotulos[nome_imagem] = 'INCERTA'

    return rotulos


def carregar_modelo(caminho_modelo):
    """Carrega o modelo CNN a partir do caminho especificado."""
    try:
        modelo = load_model(caminho_modelo)
        print("Modelo carregado com sucesso.")
        return modelo
    except Exception as e:
        print(f"Erro ao carregar o modelo: {e}")
        return None


def carregar_imagem(caminho_imagem, colorido=False):
    """Carrega uma imagem FITS ou de outro formato."""
    if caminho_imagem.endswith('.fits'):
       
        with fits.open(caminho_imagem) as hdul:
            img_data = hdul[0].data
            if img_data is None:
                return None

            
            img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data)) * 255
            img_data = img_data.astype(np.uint8)

           
            if colorido:
                img_data = cv2.cvtColor(img_data, cv2.COLOR_GRAY2BGR)

            return img_data
    else:
        try:
            img_array = plt.imread(caminho_imagem)
          
            hdu = fits.PrimaryHDU(img_array)  
            
            hdr = fits.Header()
            hdu.header = hdr

            img = hdu  

            return img
        except Exception as e:
            print(f"Erro ao carregar a imagem {caminho_imagem}: {e}")
            return None


def prever_classes(modelo, diretorio_imagens, rotulos_verdadeiros):
    """
    Prediz as classes das galáxias nas imagens usando o modelo CNN.
    """
    imagens = []
    predicoes = []
    nomes_imagens = []
    max_imagens = 9
    contador = 0

   
    threshold = 0.9 

    for imagem_nome in os.listdir(diretorio_imagens):
        if contador >= max_imagens:
            break

        caminho_imagem = os.path.join(diretorio_imagens, imagem_nome)
        img = carregar_imagem(caminho_imagem)  
        if img is not None:
            if isinstance(img, np.ndarray):
                hdu = fits.PrimaryHDU(img)
                hdr = fits.Header()
                hdu.header = hdr
                img = hdu  


            if isinstance(img, np.ndarray):
                img_resized = cv2.resize(img, (128, 128))
            else:
                img_resized = cv2.resize(img.data, (128, 128))

           
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)

           
            img_rgb = img_rgb.astype(np.float32) / 255.0

            
            if img_rgb.ndim == 3 and img_rgb.shape[2] == 3:
               
                y_center, x_center = img_rgb.shape[0] // 2, img_rgb.shape[1] // 2
                cutout = img_rgb[y_center - 32:y_center + 32, x_center - 32:x_center + 32]  
            else:
               
                img_rgb = cv2.resize(img_rgb, (64, 64))

           
            imagens.append(cutout)

            
            pred = modelo.predict(np.expand_dims(cutout, axis=0))
            probabilidade_maxima = np.max(pred)  
            # Se a probabilidade máxima for maior que o limiar, predecemos a classe
            if probabilidade_maxima > threshold:
                predicao_classe = np.argmax(pred, axis=1)[0]
            else:
                predicao_classe = -1  # Classifica como 'incerta' se abaixo do limiar

            predicoes.append(predicao_classe)
            nomes_imagens.append(imagem_nome)
            contador += 1

            # Exibe as probabilidades das classes
            print(f"Imagem: {imagem_nome}, Probabilidades: {pred[0]}, Classe Prevista: {predicao_classe}")

    
    nrows = int(np.ceil(len(imagens) / 3))
    ncols = 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))

    for i, (img, pred, nome_imagem) in enumerate(zip(imagens, predicoes, nomes_imagens)):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]

        
        ax.imshow(img, cmap='gray')  
        rotulo_verdadeiro = rotulos_verdadeiros.get(nome_imagem, "Desconhecido")
        legenda_pred = "ELÍPTICA" if pred == 1 else "ESPIRAL" if pred == 0 else "Incerta"
        ax.set_title(f'Predição: {legenda_pred}, Rótulo: {rotulo_verdadeiro}')
        ax.axis('off')

    # Desativa eixos vazios
    for j in range(len(imagens), nrows * ncols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    plt.show()

    return imagens, predicoes



def visualizar_imagens_colorida(diretorio_imagens):
    """
    Visualiza as imagens FITS em cores usando um mapa de cores.
    """
    imagens_fits = []
    rotulos = []
    max_imagens = 9
    contador = 0

    for imagem_nome in os.listdir(diretorio_imagens):
        if contador >= max_imagens:
            break

        caminho_imagem = os.path.join(diretorio_imagens, imagem_nome)
        img_fits = carregar_imagem(caminho_imagem)

        if img_fits is not None:
            imagens_fits.append(img_fits)
            rotulos.append(imagem_nome)  
            contador += 1

    n_imagens = len(imagens_fits)
    nrows = int(np.ceil(n_imagens / 3))
    ncols = 3
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))

    for i, (img_fits, rotulo) in enumerate(zip(imagens_fits, rotulos)):
        row = i // ncols
        col = i % ncols
        ax = axes[row, col]

       
        img_normalizada = cv2.normalize(img_fits, None, 0, 255, cv2.NORM_MINMAX)

       
        img_rgb = cv2.applyColorMap(img_normalizada.astype(np.uint8), cv2.COLORMAP_JET)

        ax.imshow(img_rgb)
        ax.set_title(f'Rótulo: {rotulo}')
        ax.axis('off')

   
    for j in range(len(imagens_fits), nrows * ncols):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout()
    plt.show()
    
def prever_brilho(magnitudes):  
    """
    Prediz o brilho futuro usando suavização exponencial (multi-banda).
    """
  
    previsoes = []
    for i in range(magnitudes.shape[1]):  
        modelo_brilho = ExponentialSmoothing(magnitudes[:, i], trend="add", seasonal=None)
        ajuste_brilho = modelo_brilho.fit()
        previsoes_banda = ajuste_brilho.forecast(10)
        previsoes.append(previsoes_banda)

    return np.array(previsoes).T 


def detectar_anomalias(dados):
    """
    Detecta anomalias nos dados usando Isolation Forest (multi-banda).
    """
    modelo_anomalias = IsolationForest(contamination=0.1, random_state=42)
    
  
    if dados.ndim > 1: 
        predicoes_anomalias = np.zeros(dados.shape[0])  
        for i in range(dados.shape[1]):
            dados_banda = dados[:, i].reshape(-1, 1)
            predicoes_banda = modelo_anomalias.fit_predict(dados_banda)
            predicoes_anomalias += predicoes_banda  
        predicoes_anomalias = np.sign(predicoes_anomalias) 
    else:
        dados = np.array(dados).reshape(-1, 1)
        predicoes_anomalias = modelo_anomalias.fit_predict(dados)
    
    return predicoes_anomalias


def analisar_clusters(df_features):
    """
    Analisa clusters nos dados usando KMeans e DBSCAN e visualiza os resultados.
    """
    pca = PCA(n_components=2)
    dados_pca = pca.fit_transform(df_features)

    # KMeans Clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters_kmeans = kmeans.fit_predict(dados_pca)

    # DBSCAN Clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters_dbscan = dbscan.fit_predict(dados_pca)

    fig = px.scatter(x=dados_pca[:, 0], y=dados_pca[:, 1], color=clusters_kmeans.astype(str),
                     title="Clusters de Galáxias (KMeans)", labels={"x": "PCA 1", "y": "PCA 2"})
    fig.show()

    fig = px.scatter(x=dados_pca[:, 0], y=dados_pca[:, 1], color=clusters_dbscan.astype(str),
                     title="Clusters de Galáxias (DBSCAN)", labels={"x": "PCA 1", "y": "PCA 2"})
    fig.show()


def exibir_imagens_rotulos(imagens, rotulos_verdadeiros, rotulos_previstos):
    """
    Exibe as imagens com os rótulos verdadeiros e previstos.
    """
    # Converte os rótulos verdadeiros para uma lista se necessário
    if isinstance(rotulos_verdadeiros, dict):
        rotulos_verdadeiros = [rotulos_verdadeiros.get(nome, "Desconhecido") for nome in
                               os.listdir(diretorio_imagens)[:len(imagens)]]

    # Configura o número de colunas e linhas
    num_imagens = len(imagens)
    colunas = 3
    linhas = (num_imagens // colunas) + (num_imagens % colunas > 0)

    # Cria a figura e os eixos
    fig, axes = plt.subplots(linhas, colunas, figsize=(15, 5 * linhas))

    # Desenha as imagens nos eixos
    for i, ax in enumerate(axes.flatten()):
        if i < num_imagens:
            ax.imshow(imagens[i], cmap='gray')
            ax.set_title(f"Verdadeiro: {rotulos_verdadeiros[i]}, Previsto: {rotulos_previstos[i]}")
            ax.axis('off')
        else:
            ax.axis('off')  # Desabilitar os eixos para imagens vazias

    # Adiciona uma legenda sobre os rótulos
    plt.figtext(0.5, 0.01, "0: ESPIRAL | 1: ELÍPTICA", ha="center", fontsize=12)

    plt.tight_layout()
    plt.show()


def exibir_relatorio_classificacao(rotulos_verdadeiros, rotulos_previstos):
    """
    Exibe o relatório de classificação com métricas.
    """
    from sklearn.metrics import classification_report

    # Converte os rótulos verdadeiros para uma lista se necessário
    if isinstance(rotulos_verdadeiros, dict):
        rotulos_verdadeiros = [rotulos_verdadeiros.get(nome, "Desconhecido") for nome in
                               os.listdir(diretorio_imagens)[:len(rotulos_previstos)]]

    print("Relatório de Classificação:\n")
    print(classification_report(rotulos_verdadeiros, rotulos_previstos, target_names=['ELÍPTICA', 'ESPIRAL']))

def plotar_grafico_brilho(brilhos_observados, previsoes_brilho):
    """
    Plota o gráfico do brilho observado e previsto.
    """
    n_bandas = previsoes_brilho.shape[1]
    cores = ['red', 'green', 'blue', 'purple', 'orange'] 
    plt.figure(figsize=(10, 6))  

    plt.plot(brilhos_observados, label='Brilho Observado', color='blue')

    for i in range(n_bandas):
        plt.plot(range(len(brilhos_observados), len(brilhos_observados) + len(previsoes_brilho)), 
                 previsoes_brilho[:, i], label=f'Brilho Previsto (Banda {chr(ord("G") + i)})', 
                 color=cores[i], linestyle='--', marker='o', markersize=3)  
    plt.legend()
    plt.title("Previsão do Brilho (Multi-Banda)")
    plt.xlabel("Tempo")
    plt.ylabel("Brilho")
    plt.ylim(-19, -13)  
    plt.show()
    

def executar_pipeline(arquivo_fits, diretorio_imagens, caminho_modelo_cnn, arquivo_metadados):
    """
    Executa o pipeline completo de mineração de dados.
    """
    
    modelo = carregar_modelo(caminho_modelo_cnn)
    if not modelo:
        return

    
    rotulos_reais = carregar_rotulos(arquivo_fits)

  
    imagens, predicoes = prever_classes(modelo, diretorio_imagens, rotulos_reais)
    exibir_imagens_rotulos(imagens, rotulos_reais, predicoes)

   
    visualizar_imagens_colorida(diretorio_imagens)

   
    with fits.open(arquivo_fits) as hdul:
        data = hdul[1].data
        df_metadados = pd.DataFrame(data)
        print(df_metadados.columns)

    magnitudes = df_metadados[['mag_auto_g', 'mag_auto_r', 'mag_auto_i']].values
    magnitudes = -magnitudes  

    previsoes_brilho = prever_brilho(magnitudes)

   
    plotar_grafico_brilho(magnitudes[:, 0], previsoes_brilho)  

    plt.figure()
    plt.plot(magnitudes, label='Brilho Observado')
    plt.plot(range(len(magnitudes), len(magnitudes) + len(previsoes_brilho)), previsoes_brilho, label='Brilho Previsto',
             linestyle='--')
    plt.legend()
    plt.title("Previsão do Brilho")
    plt.xlabel("Tempo")
    plt.ylabel("Brilho")
    plt.show()

    
    anomalias = detectar_anomalias(magnitudes)
    plt.figure()
    plt.plot(magnitudes[:, 0], label='Brilho (Banda G)')  # Plota a banda G como referência
    plt.scatter(df_metadados.index, magnitudes[:, 0], c=anomalias, cmap='coolwarm', label='Anomalias')  # Usa a banda G
    plt.legend()
    plt.title("Detecção de Anomalias no Brilho")
    plt.xlabel("Índice")
    plt.ylabel("Brilho")
    plt.show()

   
    analisar_clusters(df_metadados[['mag_auto_g', 'mass']])

    classes = {0: 'ESPIRAL', 1: 'ELÍPTICA'}  

   
    rotulos_previstos_str = [classes[pred] for pred in predicoes]

   
    exibir_relatorio_classificacao(rotulos_reais, rotulos_previstos_str)


arquivo_fits = 'desY1stripe82_GZ1_ES.fits'
diretorio_imagens = 'img_DESY1_stripe82_GZ1/img_DESY1_stripe82_GZ1'
caminho_modelo_cnn = 'modeloB.h5'
arquivo_metadados = 'datasets_processados/dataset_preprocessado.fits'


executar_pipeline(arquivo_fits, diretorio_imagens, caminho_modelo_cnn, arquivo_metadados)
