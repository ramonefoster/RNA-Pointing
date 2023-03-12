import pandas as pd
import csv
import numpy as np
from random import uniform
from pathlib import Path

def create_file():
    """Cria novo arquivo CSV com Header"""
    headerList = ['ah_star', 'dec_star', 'ah_scope', 'dec_scope', 'err_ah', 'err_dec', 'elevation', 'azimuth', 'temperature']

    with open(f"dataset.csv", 'w') as file:
        dw = csv.DictWriter(file, delimiter=',', 
                            fieldnames=headerList)
        dw.writeheader()

def save_dataframe(ah_star, dec_star, ah_scope, dec_scope, azimuth, elevation, temperature):
    """
    Salva dados no arquivo CSV.
    :param ah_star: Angulo Horario real da estrela
    :param dec_star: Declinacao real da estrela
    :param ah_scope: Angulo Horario do telescopio
    :param dec_scope: Declinacao do telescopio
    :param azimuth: Azimuth do telescopio
    :param elevation: Elevacao do telescopio
    :param temperatura: Temperatura interna da cupula
    """
    path_file = Path('dataset.csv')
    if not path_file.is_file():
        create_file()

    err_ah = ah_star - ah_scope
    err_dec = dec_star - dec_scope

    d = {'ah_star': [ah_star], 'dec_star': [dec_star],
        'ah_scope': [ah_scope], 'dec_scope': [dec_scope],
         'err_ah': [err_ah], 'err_dec': [err_dec], 
         'elevation':[elevation], 'azimuth': [azimuth], 'temperature': [temperature] }

    df = pd.DataFrame.from_dict(data=d)
    df.to_csv(f'dataset.csv', mode='a', index=False, header=False)

def gen_dummy_data():
    for i in range(1,100):
        ah_star = uniform(-6, 6)
        dec_star = uniform(-88.4, 43)
        #separa por quadrantes
        if ah_star < 0 and dec_star > -22.5433: 
            ah_scope = ah_star + uniform(-0.05, -0.02)
            dec_scope = dec_star + uniform(-0.16, -0.09)
        if ah_star < 0 and dec_star <= -22.5433: 
            ah_scope = ah_star + uniform(-0.03, -0.01)
            dec_scope = dec_star + uniform(0.11, 0.15)
        if ah_star >= 0 and dec_star > -22.5433: 
            ah_scope = ah_star + uniform(0.02, 0.05)
            dec_scope = dec_star + uniform(-0.16, -0.09)
        if ah_star >= 0 and dec_star <= -22.5433: 
            ah_scope = ah_star + uniform(0.01, 0.035)
            dec_scope = dec_star + uniform(0.11, 0.07)
        temperature = uniform(4, 16)
        azimuth, elevation = calcAzimuthAltura(ah_scope, dec_scope)
        if elevation > 10:
            save_dataframe(ah_star, dec_star, ah_scope, dec_scope, azimuth, elevation, temperature)

def calcAzimuthAltura(ah, dec):
    """
    Calcula Azimuth e Elevacao
    :Param ah: Angulo Horario do alvo ()
    :Param dec: Declinacao do alvo 
    :Param temp: Temperatura Graus Centigrados
    :return: Azimuth e Elevacao (em graus)
    """
    DEG = 180 / np.pi
    RAD = np.pi / 180.0
    H = ah * 15
    latitude = -22.534

    sinAltitude = (np.sin(dec * RAD)) * (np.sin(latitude * RAD)) + (np.cos(dec * RAD) * np.cos(latitude * RAD) * np.cos(H * RAD))
    elevation = np.arcsin(sinAltitude) * DEG 

    y = -1 * np.sin(H * RAD)
    x = (np.tan(dec * RAD) * np.cos(latitude * RAD)) - (np.cos(H * RAD) * np.sin(latitude * RAD))

    azimuth = np.arctan2(y, x) * DEG

    if (azimuth < 0) :
        azimuth = azimuth + 360

    return(azimuth, elevation)

create_file()
gen_dummy_data()
