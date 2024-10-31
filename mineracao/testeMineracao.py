import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model

# Função para carregar os rótulos reais do arquivo FITS
def carregar_rotulos(arquivo_fits):
    with fits.open(arquivo_fits) as hdul:
        dados = hdul[1].data
        rotulos_espiral = dados['SPIRAL']
        rotulos_eliptica = dados['ELLIPTICAL']
        rotulos_incertos = dados['UNCERTAIN']
        
        # Define rótulos reais: 1 para espiral, 0 para elíptica, -1 para incerto
        rotulos_reais = np.where(rotulos_espiral == 1, 1, np.where(rotulos_eliptica == 1, 0, -1))
    
    return rotulos_reais

# Função para carregar as imagens FITS de um diretório
def carregar_imagens(diretorio_imagens):
    imagens = []
    arquivos = sorted(os.listdir(diretorio_imagens))  # Ordena para manter a correspondência com os rótulos
    for arquivo in arquivos:
        if arquivo.endswith('.fits'):
            caminho_imagem = os.path.join(diretorio_imagens, arquivo)
            with fits.open(caminho_imagem) as hdul:
                imagem = hdul[0].data  # Carrega a imagem da galáxia
                imagens.append(imagem)
    
    return np.array(imagens), arquivos

# Função para classificar galáxias usando o modelo de CNN treinado
def classificar_galaxias(imagens, modelo_cnn):
    imagens_expandidas = np.expand_dims(imagens, axis=-1)  # Adiciona o canal extra (se necessário)
    predicoes = modelo_cnn.predict(imagens_expandidas)
    predicoes = np.argmax(predicoes, axis=1)  # Classifica entre 0 (elíptica) e 1 (espiral)
    return predicoes

# Função para analisar o desempenho do modelo
def analisar_desempenho(predicoes, rotulos_reais):
    print("Matriz de Confusão:")
    print(confusion_matrix(rotulos_reais, predicoes))
    
    print("\nRelatório de Classificação:")
    print(classification_report(rotulos_reais, predicoes, target_names=['Elíptica', 'Espiral']))
    
    # Visualização gráfica da matriz de confusão
    plt.figure(figsize=(6,6))
    matriz_confusao = confusion_matrix(rotulos_reais, predicoes)
    plt.imshow(matriz_confusao, cmap='Blues', interpolation='none')
    plt.title('Matriz de Confusão')
    plt.colorbar()
    plt.xticks([0, 1], ['Elíptica', 'Espiral'])
    plt.yticks([0, 1], ['Elíptica', 'Espiral'])
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.show()

# Função para aplicar DBSCAN e encontrar clusters
def encontrar_clusters(imagens, eps=0.5, min_samples=5):
    pca = PCA(n_components=50)  # Reduz a dimensionalidade para análise
    imagens_flat = imagens.reshape(imagens.shape[0], -1)
    imagens_pca = pca.fit_transform(imagens_flat)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(imagens_pca)
    return clusters

# Função para detectar anomalias usando Isolation Forest
def detectar_anomalias(imagens, contaminação=0.05):
    iforest = IsolationForest(contamination=contaminação)
    imagens_flat = imagens.reshape(imagens.shape[0], -1)
    anomalias = iforest.fit_predict(imagens_flat)
    return anomalias

# Função para exibir clusters ou anomalias
def exibir_clusters(imagens, clusters, titulo="Clusters Detectados"):
    unique_clusters = np.unique(clusters)
    plt.figure(figsize=(15, 15))
    for i, cluster in enumerate(unique_clusters):
        plt.subplot(5, 5, i + 1)
        indices = np.where(clusters == cluster)[0]
        plt.imshow(imagens[indices[0]], cmap='gray')  # Exibe uma imagem representando o cluster
        plt.title(f"Cluster {cluster}")
        plt.axis('off')
    plt.suptitle(titulo, fontsize=16)
    plt.show()

# Função para salvar as imagens tratadas em outro diretório
def salvar_imagens_tratadas(imagens, arquivos, predicoes, diretorio_destino="imagens_tratadas"):
    if not os.path.exists(diretorio_destino):
        os.makedirs(diretorio_destino)
    
    for i, arquivo in enumerate(arquivos):
        predicao_str = 'espiral' if predicoes[i] == 1 else 'eliptica'
        caminho_destino = os.path.join(diretorio_destino, f"{predicao_str}_{arquivo}.png")
        plt.imsave(caminho_destino, imagens[i], cmap='gray')

# Função para carregar e analisar os metadados do arquivo dataset_preprocessado.fits
def analisar_metadados(arquivo_metadados):
    with fits.open(arquivo_metadados) as hdul:
        dados = hdul[1].data
        df_metadados = pd.DataFrame(np.array(dados).byteswap().newbyteorder())  # Converte os metadados para um DataFrame
    
    print("\nResumo Estatístico dos Metadados:")
    print(df_metadados.describe())
    
    # Plotar histogramas de algumas variáveis importantes
    variaveis_importantes = ['MAG_AUTO_G', 'MAG_AUTO_R', 'Z_PHOTO']  # Exemplo de variáveis
    for variavel in variaveis_importantes:
        plt.figure()
        df_metadados[variavel].hist(bins=50)
        plt.title(f"Distribuição de {variavel}")
        plt.xlabel(variavel)
        plt.ylabel('Frequência')
        plt.show()

    # Análise de correlação entre variáveis
    plt.figure(figsize=(10, 8))
    corr = df_metadados.corr()
    plt.imshow(corr, cmap='coolwarm', interpolation='none')
    plt.colorbar()
    plt.title('Matriz de Correlação entre Metadados')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()

    return df_metadados

# Função para rodar todo o pipeline
def executar_pipeline(diretorio_imagens, arquivo_rotulos, caminho_modelo_cnn, arquivo_metadados):
    # Carregar os rótulos reais
    rotulos_reais = carregar_rotulos(arquivo_rotulos)
    
    # Carregar as imagens do diretório
    imagens, arquivos = carregar_imagens(diretorio_imagens)
    
    # Carregar o modelo treinado
    modelo_cnn = load_model(caminho_modelo_cnn)
    
    # Classificar as imagens usando o modelo treinado
    predicoes = classificar_galaxias(imagens, modelo_cnn)
    
    # Salvar as imagens tratadas
    salvar_imagens_tratadas(imagens, arquivos, predicoes, diretorio_destino="imagens_tratadas")
    
    # Analisar o desempenho
    analisar_desempenho(predicoes, rotulos_reais)
    
    # Detectar clusters usando DBSCAN
    clusters = encontrar_clusters(imagens)
    exibir_clusters(imagens, clusters, titulo="Clusters Encontrados com DBSCAN")
    
    # Detectar anomalias usando Isolation Forest
    anomalias = detectar_anomalias(imagens)
    exibir_clusters(imagens, anomalias, titulo="Anomalias Detectadas com Isolation Forest")
    
    # Analisar os metadados
    analisar_metadados(arquivo_metadados)


# Caminhos dos arquivos
diretorio_imagens = 'img_DESY1_stripe82_GZ1/img_DESY1_stripe82_GZ1'  # Insira o caminho correto para o diretório de imagens
arquivo_rotulos = 'desY1stripe82_GZ1_ES.fits'  # Insira o caminho correto para o arquivo de rótulos
caminho_modelo_cnn = 'modelo_galaxia.h5'  # Insira o caminho correto para o seu modelo treinado
arquivo_metadados = 'datasets_processados/dataset_preprocessado.fits'

# Executar o pipeline
executar_pipeline(diretorio_imagens, arquivo_rotulos, caminho_modelo_cnn, arquivo_metadados)
