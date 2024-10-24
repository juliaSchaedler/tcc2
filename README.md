# TCC: Universo em Bits: Mineração e Análise de Dados Aplicado à Astronomia para Detecção de Galáxias Usando Redes Neurais Convolucionais

Objetivo: Aprimorar algoritmos de redes neurais convolucionais e aplicá-lo nas técnicas de mineração de dados astronômicos, para detecção de galáxias, para melhorar o desempenho e obter melhores resultados comparados com aqueles já encontrados em bases de dados analisadas da DES Project e Galaxy Zoo, além de realizar novas aplicações em bases de dados ainda não analisadas.

## Etapas:
### Separação de dados:

O dataset escolhido é o catálogo DES Y3 GOLD, disponibilizado por Cheng et al. 2021 no DES Project. O link para ele está aqui: 
https://des.ncsa.illinois.edu/releases/y3a2/gal-morphology

### Pré-processamento dos dados:


#### Carregamento e conversão de dados

* **Carregamento de dados FITS:** Carrega os dados do arquivo FITS especificado e os converte em um DataFrame pandas para facilitar a manipulação.
* **Exibição de informações:** Exibe o número de entradas e as colunas disponíveis no DataFrame.

#### Geração de rótulos

* **Criação de rótulos:** Gera rótulos para cada galáxia com base nos valores das colunas `pE` (probabilidade de ser elíptica) e `pS` (probabilidade de ser espiral).
    * Se `pE` > 0.5, a galáxia é rotulada como "elíptica".
    * Se `pS` > 0.5, a galáxia é rotulada como "espiral".
    * Caso contrário, a galáxia é rotulada como "desconhecido".
* **Seleção de colunas:** Seleciona as colunas `DES_Y3_ID` (ID da galáxia) e `rotulo` para o DataFrame de rótulos.

#### Normalização de dados

* **Normalização de magnitudes:** Normaliza as colunas de magnitude (`MAG_AUTO_G`, `MAG_AUTO_R`, `MAG_AUTO_I`, `MAG_AUTO_Z`, `MAG_AUTO_Y`) usando a padronização Z-score (subtraindo a média e dividindo pelo desvio padrão).
* **Normalização de erros:** Normaliza as colunas de erro de magnitude (`MAGERR_AUTO_G`, `MAGERR_AUTO_R`, `MAGERR_AUTO_I`, `MAGERR_AUTO_Z`, `MAGERR_AUTO_Y`) usando a padronização Z-score.

#### Pré-processamento em paralelo

* **Divisão de dados:** Divide o DataFrame em partes para processamento paralelo.
* **Processamento paralelo:** Utiliza multiprocessing para normalizar as colunas de magnitude e erro em paralelo, acelerando o processo.
* **Concatenação de resultados:** Concatena as partes normalizadas em um único DataFrame.

#### Salvamento dos dados

* **Criação de arquivo FITS:** Salva o DataFrame pré-processado em um novo arquivo FITS.
* **Conversão de tipos de dados:** Converte os tipos de dados das colunas para formatos compatíveis com FITS.
* **Salvamento dos rótulos:** Salva o DataFrame de rótulos em um arquivo FITS separado.


### Treinamento e avaliação da CNN

#### Pré-processamento das imagens de galáxias

* **Carregamento de imagens FITS:** Carrega imagens FITS de um diretório especificado, convertendo-as para o formato RGB e redimensionando-as para 64x64 pixels.
* **Aumento de dados:** Aplica técnicas de aumento de dados como rotação, translação, cisalhamento, zoom e espelhamento horizontal para aumentar a variabilidade do conjunto de dados e melhorar a generalização do modelo.
* **Carregamento de rótulos e metadados:** Carrega os rótulos (espiral, elíptica, incerta) e metadados (ra, dec, z, etc.) de um arquivo FITS separado.
* **Remoção de amostras incertas:** Remove as amostras com rótulo "incerta" do conjunto de dados.
* **Conversão de rótulos:** Converte os rótulos para um formato binário (0 para espiral, 1 para elíptica).
* **Balanceamento de classes:** Utiliza a técnica de oversampling para balancear as classes, garantindo que o modelo não seja enviesado para a classe majoritária.
* **Divisão em treino e teste:** Divide o conjunto de dados em conjuntos de treinamento e teste.

#### Construção e treinamento do modelo

* **Criação do modelo:** Define uma CNN com três camadas convolucionais, cada uma seguida por batch normalization, max pooling e dropout. A camada de saída utiliza a função de ativação softmax para classificação binária.
* **Compilação do modelo:** Compila o modelo com o otimizador Adam, a função de perda sparse categorical crossentropy e a métrica de acurácia.
* **Treinamento do modelo:** Treina o modelo com os dados de treinamento, utilizando um callback ModelCheckpoint para salvar o melhor modelo durante o treinamento.
* **Monitoramento do tempo de treinamento:** Utiliza um callback personalizado (TimeHistory) para registrar o tempo de treinamento de cada época.

#### Avaliação do modelo

* **Predição:** Realiza previsões nos dados de teste.
* **Matriz de confusão:** Plota a matriz de confusão para visualizar o desempenho do modelo em cada classe.
* **Métricas de avaliação:** Calcula e imprime o relatório de classificação (precision, recall, f1-score, support) para cada classe.
* **Gráficos de avaliação:** Plota gráficos da perda e acurácia durante o treinamento para analisar o processo de aprendizado.

#### Observações

* O código utiliza GPU para acelerar o treinamento, se disponível.
* O código inclui otimizações de desempenho, como o uso de mixed precision.
* O código é modularizado em funções para facilitar a leitura e manutenção.

## Bibliotecas utilizadas
Bibliotecas usadas durante todo o processo:
* `numpy`: para manipulação numérica.
* `tensorflow`: para construir e treinar o modelo de deep learning.
* `logging`: para registrar mensagens de log.
* `matplotlib.pyplot`: para plotar gráficos.
* `tqdm`: para exibir barras de progresso.
* `sklearn`: para pré-processamento de dados e métricas de avaliação.
* `astropy.io.fits`: para ler arquivos FITS.
* `cv2`: para processamento de imagens.
* `glob`: para encontrar arquivos.
* `os`: para interagir com o sistema operacional.
* `imblearn`: para balanceamento de classes.
* `tensorflow.keras`: para construir e treinar o modelo.


#### Requisitos:
- Python 3.x
- Instalar bibliotecas necessárias (listadas acima)
- Utilizar ambientes virtuais (venv ou conda, opcional)
- **Mais informações, ler o info.txt**

