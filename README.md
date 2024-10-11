# tcc2
Desenvolvimento da parte 2 do meu TCC para o curso de ciência da computação. O assunto é mineração de dados astronômicos, utilizando redes neurais convolucionais.

## Etapas:
### Separação de dados:

O dataset escolhido é o catálogo DES Y3 GOLD, disponibilizado por Cheng et al. 2021 no DES Project. O link para ele está aqui: 
https://des.ncsa.illinois.edu/releases/y3a2/gal-morphology

### Pré-processamento dos dados:


Este script realiza o pré-processamento de dados de galáxias contidos em um arquivo no formato `.fits`. Ele é utilizado para preparar dados de galáxias espirais e elípticas, removendo valores vazios, normalizando colunas específicas, tratando outliers e criando rótulos para classificação binária (galáxias espirais e elípticas).

#### Funcionalidades principais:

1. **Carregamento do arquivo FITS**: Os dados são carregados e convertidos para o formato "little-endian" se necessário.
2. **Remoção de valores vazios**: Linhas com valores ausentes são removidas.
3. **Normalização de dados**: As colunas de interesse (pS, pE, magnitude, etc.) são normalizadas para o intervalo [0, 1].
4. **Tratamento de outliers**: Remoção de valores considerados outliers (baseado no z-score).
5. **Criação de rótulos**: Com base na probabilidade `pS` (probabilidade de ser uma galáxia espiral), o script cria rótulos binários:
   - `1` para galáxias espirais (`pS >= 0.8`)
   - `0` para galáxias elípticas.
6. **Exportação de arquivos**:
   - **Dados processados**: São salvos no arquivo `processed_galaxies.csv` sem a coluna de rótulos.
   - **Rótulos**: São salvos separadamente no arquivo `galaxy_labels.csv`.

#### Uso:

Basta rodar o script `preprocessamento.py` para gerar dois arquivos CSV:
- `processed_galaxies.csv`: Contém os dados limpos e normalizados.
- `galaxy_labels.csv`: Contém os rótulos de classificação das galáxias.

#### Requisitos:
- Python 3.x
- Bibliotecas: `pandas`, `numpy`, `astropy`, `scipy`
- **Mais informações, ler o info.txt**

