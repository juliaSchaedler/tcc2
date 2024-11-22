import numpy as np
import pandas as pd
from astropy.io import fits
import logging
from scipy import stats

# Configurando o logging para monitoramento
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Função para carregar o arquivo .fits e converter para little-endian
def load_fits_file(file_path):
    logging.info(f'Carregando arquivo {file_path}')
    with fits.open(file_path) as hdul:
        data = hdul[1].data
        columns = hdul[1].columns.names
        
        # Convertendo para little-endian se necessário
        data = np.array(data, dtype=data.dtype.newbyteorder('<'))
        
    logging.info(f'Data shape: {data.shape}')
    logging.info(f'Colunas disponíveis: {columns}')
    return data, columns

# Função para remover valores vazios
def remove_missing_values(data):
    logging.info('Removendo valores vazios...')
    df = pd.DataFrame(data)
    df_clean = df.dropna()
    logging.info(f'Valores vazios removidos: {len(df) - len(df_clean)}')
    return df_clean

# Função para normalizar os dados 
def normalize_data(df, columns):
    logging.info('Normalizando dados...')
    cols_to_normalize = ['pE', 'pS', 'MAG_AUTO_I', 'DNF_ZMEAN_SOF']
    
    # Verifica se as colunas estão presentes antes de normalizar
    cols_to_normalize = [col for col in cols_to_normalize if col in columns]
    if cols_to_normalize:
        df[cols_to_normalize] = df[cols_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    else:
        logging.warning('Nenhuma das colunas para normalização foi encontrada.')
    
    return df

# Função para tratar outliers
def handle_outliers(df, columns):
    logging.info('Tratando outliers...')
    cols_to_check = ['pE', 'pS', 'MAG_AUTO_I', 'DNF_ZMEAN_SOF']
    
    # Verifica se as colunas estão presentes antes de verificar outliers
    cols_to_check = [col for col in cols_to_check if col in columns]
    
    if cols_to_check:
        z_scores = np.abs(stats.zscore(df[cols_to_check]))
        df_clean = df[(z_scores < 3).all(axis=1)]
        logging.info(f'Outliers removidos: {len(df) - len(df_clean)}')
    else:
        logging.warning('Nenhuma das colunas para verificação de outliers foi encontrada.')
        df_clean = df
    
    return df_clean

# Função para criar rótulos com base no valor de pS (probabilidade de ser espiral)
def create_labels(df, threshold=0.8):
    logging.info(f'Criando rótulos com base no limiar {threshold} para a coluna pS...')
    
    if 'pS' in df.columns:
        # Criar rótulos: 1 para Espiral, 0 para Elíptica
        labels = np.where(df['pS'] >= threshold, 1, 0)
        logging.info('Rótulos criados com sucesso.')
        return labels
    else:
        logging.error('A coluna pS não foi encontrada no dataframe.')
        return None

# Função principal de pré-processamento
def preprocess_data(file_path):
    data, columns = load_fits_file(file_path)
    
    # Remover valores vazios
    df_clean = remove_missing_values(data)
    
    # Normalizar os dados
    df_normalized = normalize_data(df_clean, columns)
    
    # Tratar outliers
    df_no_outliers = handle_outliers(df_normalized, columns)
    
    # Criar rótulos (Espiral ou Elíptica) com base em pS
    labels = create_labels(df_no_outliers)
    
    logging.info('Processamento finalizado.')
    return df_no_outliers, labels

# Exemplo de uso
file_path = 'dataset.fits'
df_processed, labels = preprocess_data(file_path)

# Salvar os dados processados sem rótulos
df_processed.to_csv('processed_galaxies.csv', index=False)
logging.info('Dados processados salvos em arquivo.')

# Salvar os rótulos separadamente
if labels is not None:
    pd.DataFrame(labels, columns=['label']).to_csv('galaxy_labels.csv', index=False)
    logging.info('Rótulos salvos em arquivo separado.')
