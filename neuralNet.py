import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

dataframe = pd.read_csv("dataset.csv")

X = dataframe[["azimuth", "elevation"]]
Y = dataframe["err_ah"]
# X = X.values.reshape(-1,1)

#Normalizacao
escala = StandardScaler()
escala.fit(X)
X_norm = escala.transform(X)

# #divide em treinamento e teste
X_norm_train, X_norm_test, y_train, y_teste = train_test_split(X_norm, Y, test_size=0.2)

# #Processamento
rna = MLPRegressor(hidden_layer_sizes=(10,5), max_iter=1000, tol=0.0000001, 
                   learning_rate_init=0.1, solver="sgd", activation="logistic", 
                   learning_rate="constant", verbose=2)

rna.fit(X_norm_train, y_train)

Y_rna_previsao = rna.predict(X_norm_test)

#PREVISAO
X_futuro = np.array([[85,55]])
X_futuro_norm = escala.transform(X_futuro)

Y_rna_prever_futuro = rna.predict(X_futuro_norm)
print("Erro AH previsto: ", Y_rna_prever_futuro)



# plt.scatter(X, Y)
# plt.xlabel("Azimuth (deg)")
# plt.ylabel("Erro em AH (minarc)")
# plt.title("Relacao Erro Ah e Azimuth")
# plt.show()

