import multiprocessing
import logging
from memory_profiler import profile

if __name__ == '__main__':
    multiprocessing.freeze_support()

    import dask
    import dask.dataframe as dd
    from dask.distributed import Client
    import numpy as np
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.utils import to_categorical
    from sklearn.preprocessing import MinMaxScaler
    from dask.diagnostics import progress
    from astropy.io import fits

    # Configurar o logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Iniciar o cliente Dask com otimizações e ajuste de memória
    client = Client(n_workers=2, memory_limit='2GB')  

    # Caminho para o arquivo FITS
    fits_file_path = 'dataset.fits'

    # Função para ler e pré-processar um HDU do arquivo FITS
    @profile
    def preprocess_hdu(fits_file_path, hdu_index):
        # 1. Ler apenas as colunas necessárias com chunksize menor
        with fits.open(fits_file_path, columns=['pE', 'pS', 'MAG_AUTO_G', 'MAG_AUTO_R', 'MAG_AUTO_I', 'DNF_ZMEAN_MOF', 'confidence_flag'], chunksize=1000) as hdul:
            hdu = hdul[hdu_index]
            if isinstance(hdu, fits.BinTableHDU):
                data = hdu.data
                df = dd.from_array(data, columns=hdu.columns.names)

                # 2. Lidar com valores ausentes (substituir pela média)
                means = df.mean(numeric_only=True).compute()
                df = df.fillna(means)

                # 3. Normalizar as features MAG_AUTO_* (diretamente no Dask DataFrame)
                scaler = MinMaxScaler()
                cols_to_normalize = ['MAG_AUTO_G', 'MAG_AUTO_R', 'MAG_AUTO_I', 'MAG_AUTO_Z', 'MAG_AUTO_Y']
                df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize].to_dask_array(lengths=True))
                del scaler  # Libera a memória ocupada pelo scaler

                # 4. Remover features irrelevantes (incluindo DNF_ZMEAN_MOF)
                df = df[['pE', 'pS', 'MAG_AUTO_G', 'MAG_AUTO_R', 'MAG_AUTO_I', 'DNF_ZMEAN_MOF']]

                return df

    # Criar uma lista de tarefas delayed para cada HDU
    delayed_tasks = [dask.delayed(preprocess_hdu)(fits_file_path, i) for i in range(1, len(fits.open(fits_file_path)))]

    # Executar as tarefas em paralelo com o Dask, limitando o número de workers
    with progress.ProgressBar():
        results = dask.compute(*delayed_tasks, num_workers=2)

    # Concatenar os resultados em um único DataFrame
    ddf = dd.concat(results)

    # Filtrar as galáxias com base na confiança
    ddf = ddf[ddf['confidence_flag'] > 0.8]

    # Criar rótulos (usando limiar de 0.8)
    print("Criando os rótulos...")
    with progress.ProgressBar():
        rotulos = ddf['pS'].map(lambda x: 1 if x >= 0.8 else 0).compute()
    print("Criação dos rótulos concluída.")

    # One-hot encoding dos rótulos
    rotulos = to_categorical(rotulos)

    # Converter as features para um array NumPy
    print("Convertendo as features para array NumPy...")
    with progress.ProgressBar():
        features = ddf.to_numpy()
    print("Conversão concluída.")

    del ddf  # Liberar memória

    # Dividir em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(features, rotulos, test_size=0.2, random_state=42)

    # Salvar as features pré-processadas em um arquivo CSV
    print("Salvando as features pré-processadas...")
    features_df = pd.DataFrame(X_train)
    features_df.to_csv('features_preprocessadas.csv', index=False)
    print("Features salvas em 'features_preprocessadas.csv'.")

    # Salvar os rótulos em um arquivo CSV
    print("Salvando os rótulos...")
    rotulos_df = pd.DataFrame(y_train)
    rotulos_df.to_csv('rotulos.csv', index=False)
    print("Rótulos salvos em 'rotulos.csv'.")
