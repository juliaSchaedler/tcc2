from astropy.io import fits
import os

# Substitua 'seu_arquivo.fits' pelo caminho para o seu arquivo FITS
with fits.open('desY1stripe82_GZ1_ES.fits') as hdul:
    data = hdul[1].data  # Assumindo que os dados estão na primeira extensão HDU

    # Conta a quantidade de dados não nulos em cada coluna
    n_spiral = len(data['SPIRAL'])
    n_elliptical = len(data['ELLIPTICAL'])
    n_uncertain = len(data['UNCERTAIN'])

    print(f'Número de dados na coluna SPIRAL: {n_spiral}')
    print(f'Número de dados na coluna ELLIPTICAL: {n_elliptical}')
    print(f'Número de dados na coluna UNCERTAIN: {n_uncertain}')

# Substitua 'caminho/para/o/diretorio' pelo caminho para o diretório desejado
diretorio = 'img_DESY1_stripe82_GZ1/img_DESY1_stripe82_GZ1'
num_arquivos = len(os.listdir(diretorio))

print(f'Número de arquivos no diretório: {num_arquivos}')
