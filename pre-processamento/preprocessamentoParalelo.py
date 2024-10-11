import pandas as pd
import numpy as np
from astropy.io import fits
import multiprocessing
import logging

# Configuração do logging
logging.basicConfig(level=logging.INFO)

# Função para carregar o arquivo FITS e converter em DataFrame
def carregar_fits_para_dataframe(fits_path):
    logging.info(f"Carregando arquivo FITS: {fits_path}")
    with fits.open(fits_path) as hdul:
        data = hdul[1].data
        df = pd.DataFrame(np.array(data).byteswap().newbyteorder())
    logging.info(f"Verificando conteúdo do arquivo FITS:")
    logging.info(f"Número de entradas: {len(df)}")
    logging.info(f"Colunas disponíveis: {list(df.columns)}")
    return df

# Função para salvar dados CSV
def salvar_csv(df, csv_path):
    logging.info(f"Salvando dados no arquivo CSV: {csv_path}")
    df.to_csv(csv_path, index=False)
    logging.info(f"Primeiras 5 linhas do arquivo CSV salvo:\n{df.head()}")
    logging.info(f"Número de linhas no arquivo CSV salvo: {len(df)}")

# Função para criar rótulos baseados em pE e pS
def gerar_rotulos(df):
    logging.info("Gerando rótulos baseados em pE e pS")
    conditions = [
        (df['pE'] > 0.5),
        (df['pS'] > 0.5)
    ]
    choices = ['eliptica', 'espiral']
    df['rotulo'] = np.select(conditions, choices, default='desconhecido')
    return df[['DES_Y3_ID', 'rotulo']]

# Função para normalizar as colunas de magnitude e erros
def normalizar_colunas(df):
    logging.info("Normalizando colunas de magnitude e erros")
    colunas_mag = ['MAG_AUTO_G', 'MAG_AUTO_R', 'MAG_AUTO_I', 'MAG_AUTO_Z', 'MAG_AUTO_Y']
    colunas_magerr = ['MAGERR_AUTO_G', 'MAGERR_AUTO_R', 'MAGERR_AUTO_I', 'MAGERR_AUTO_Z', 'MAGERR_AUTO_Y']
    
    df[colunas_mag] = df[colunas_mag].apply(lambda x: (x - x.mean()) / x.std())
    df[colunas_magerr] = df[colunas_magerr].apply(lambda x: (x - x.mean()) / x.std())
    
    return df

# Função para pré-processar os dados em paralelo
def preprocessar_dados_em_paralelo(df, n_processos=4):
    logging.info(f"Pré-processando dados com {n_processos} processos paralelos")
    tamanho_particao = len(df) // n_processos
    partes = [df.iloc[i * tamanho_particao: (i + 1) * tamanho_particao] for i in range(n_processos)]
    
    with multiprocessing.Pool(processes=n_processos) as pool:
        partes_normalizadas = pool.map(normalizar_colunas, partes)
    
    df_normalizado = pd.concat(partes_normalizadas)
    return df_normalizado

# Função para testar o processo completo
def executar_processo_completo(fits_file, csv_file, rotulos_file):
    # Carregar e salvar os dados FITS para CSV
    df = carregar_fits_para_dataframe(fits_file)
    salvar_csv(df, csv_file)

    # Gerar e salvar os rótulos
    df_rotulos = gerar_rotulos(df)
    salvar_csv(df_rotulos, rotulos_file)

    # Pré-processar dados usando paralelismo
    df_preprocessado = preprocessar_dados_em_paralelo(df)
    salvar_csv(df_preprocessado, f'preprocessado_{csv_file}')
    logging.info("Processamento completo finalizado.")

# Execução do script
if __name__ == '__main__':
    fits_file = 'dataset.fits'
    csv_file = 'testeProcessado.csv'
    rotulos_file = 'testeRotulos.csv'
    executar_processo_completo(fits_file, csv_file, rotulos_file)
