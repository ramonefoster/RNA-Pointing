import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pickle
import joblib

def read_data():
    dataframe = pd.read_csv("dataset.csv")

    X = dataframe[["azimuth", "elevation", "ah_star", "dec_star", "temperature"]]
    Y = dataframe[["err_ah", "err_dec"]]
    # X = X.values.reshape(-1,1)

    return X, Y

def save_train_test():
    #Normalizacao
    X, Y = read_data()
    escala = StandardScaler()
    escala.fit(X)
    X_norm = escala.transform(X)

    # #divide em treinamento e teste
    X_norm_train, X_norm_test, y_train, y_teste = train_test_split(X_norm, Y, test_size=0.2)

    with open('obs_train_test.pkl', mode='wb') as f:
        pickle.dump([X_norm_train, X_norm_test, y_train, y_teste], f)
   

def train():
    rna = MLPRegressor(hidden_layer_sizes=(200, 200), max_iter=2000, tol=0.0000001, 
                    learning_rate_init=0.01, solver="adam", activation="relu", 
                    learning_rate="constant", verbose=2)    

    with open('obs_train_test.pkl', mode='rb') as f:
        X_norm_train, X_norm_test, y_train, y_teste = pickle.load(f)

    # #Processamento    
    rna.fit(X_norm_train, y_train)
    #save trained model to file
    # with open('NNmodel.pkl', mode='wb') as f:
    #     pickle.dump([rna], f)
    joblib.dump(rna, 'NNmodel.pkl')

    Y_rna_previsao = rna.predict(X_norm_test)

    r2_rna = r2_score(y_teste, Y_rna_previsao)
    print("Aproximacao dos dados reais:")
    print(r2_rna)

def make_predict(ah=None, dec=None, temp=None ):
    rna = joblib.load('NNmodel.pkl')

    #PREVISAO
    az, elevation = calcAzimuthAltura(ah,dec)
    X_futuro = np.array([[az, elevation, ah, dec, temp]])
    X, Y = read_data()
    escala = StandardScaler()
    escala.fit(X)
    X_futuro_norm = escala.transform(X_futuro)

    Y_rna_prever_futuro = rna.predict(X_futuro_norm)
    fator_correct_ah = Y_rna_prever_futuro[0][0]
    fator_correct_dec = Y_rna_prever_futuro[0][1]
    print("AH SCOPE: ", ah, "DEC SCOPE: ", dec)
    print("CORRECTED COORDINATES: ")
    print("AH: ", ah+fator_correct_ah, "DEC:", dec+fator_correct_dec)

    # plt.scatter(X, Y)
    # plt.xlabel("Azimuth (deg)")
    # plt.ylabel("Erro em AH (minarc)")
    # plt.title("Relacao Erro Ah e Azimuth")
    # plt.show()

def calcAzimuthAltura(ah, dec):
    """Calculates Azimuth and Zenith"""
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

#save_train_test()
#train()
make_predict(-1.375, -53.7256, 12)
