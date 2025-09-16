import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm 
import torch.nn as nn
import torch.nn.functional as F

# Figura global con dos subplots: uno para scatter y otro para la curva de error
fig2, ax1 = plt.subplots(1,2, figsize=(7,3))


class SimpleNeuralNetwork(nn.Module):
    """
    Red neuronal simple para regresión.

    Arquitectura:
    - Capa totalmente conectada con 13 entradas y 8 salidas.
    - Capa de salida con 1 neurona (predicción continua).
    """

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features=13, out_features=8)
        self.fc2 = nn.Linear(in_features=8, out_features=1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Realiza la propagación hacia adelante.

        Parámetros
        ----------
        x : torch.tensor
            Tensor de entrada.

        Retorna
        -------
        torch.tensor
            Predicción continua de la red.
        """
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def analisisDeDatos(df: pd.DataFrame) -> None:
    """
    Muestra histogramas para variables numéricas clave.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con las columnas ['unit_price', 'total_price', 'quantity', 'reward_points'].
    """
    fig, ax = plt.subplots(2,2, figsize=(7,3))
    columns = ['unit_price', 'total_price', 'quantity', 'reward_points']
    
    for i in tqdm(range(len(columns))):
        if i > 1:
            ax[1, i-2].hist(df[columns[i]], bins=40, edgecolor='k',
                            label=f'{columns[i]} MODE{df[columns[i]].mode()}')
            ax[1, i-2].legend(title='Leyenda:')
        else:
            ax[0, i].hist(df[columns[i]], bins=40, edgecolor='k',
                          label=f'{columns[i]} MODE{df[columns[i]].mode()}')
            ax[0,i].legend(title='Leyenda:')        


def normalizacion (df: pd.DataFrame, colsINDEX: dict) -> pd.DataFrame:
    """
    Convierte variables categóricas en numéricas mediante codificación.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con columnas a transformar.
    colsINDEX : dict
        Diccionario con columnas y tipo de codificación:
        - 'cat' : codificación con códigos numéricos.
        - 'one' : one-hot encoding.

    Retorna
    -------
    pd.DataFrame
        DataFrame transformado.
    """
    for i in tqdm(range(len(df.columns))):
        for index, key in enumerate(colsINDEX):
            if i == index:
                if colsINDEX[key] == 'cat':
                    df[key] = df[key].astype('category')
                    df[key] = df[key].cat.codes
                else:
                    df.loc[:,key].astype('category')
                    df = pd.get_dummies(df, columns=[key], prefix=f'{key}')
    return df


def estandarizacion(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Estandariza variables numéricas (media=0, desviación estándar=1).

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame original.
    columns : list
        Lista de columnas a estandarizar.

    Retorna
    -------
    pd.DataFrame
        DataFrame con las columnas estandarizadas.
    """
    new_df = df[columns]
    new_df = (new_df - new_df.mean()) / new_df.std() 
    df.drop(columns=columns, inplace=True)
    df = pd.concat([df, new_df], axis=1).astype(float)
    return df  


def aleatorizacion_de_datos(t: torch.tensor, columns_target: list, percents: float) -> tuple:
    """
    Baraja y divide datos en entrenamiento y prueba.

    Parámetros
    ----------
    t : torch.tensor
        Tensor con datos completos.
    columns_target : list
        Índices de columnas objetivo (target).
    percents : float
        Porcentaje de datos para entrenamiento (ej. 0.8).

    Retorna
    -------
    tuple
        (x_train, x_test, y_train, y_test) como tensores de PyTorch.
    """
    index = torch.randperm(t.shape[0])
    t = t[index]

    x = t[:, [i for i in tqdm(range(t.shape[1])) if i not in columns_target]]
    y = t[:, [i for i in tqdm(range(t.shape[1])) if i in columns_target]]

    x_train, x_test = torch.split(x,int(x.shape[0]*percents), dim=0)
    y_train, y_test  = torch.split(y,int(y.shape[0]*percents), dim=0)
    return x_train, x_test, y_train, y_test


def main() -> None:
    """
    Pipeline principal:
    1. Carga y preprocesa los datos (log, normalización, estandarización).
    2. Divide en entrenamiento y prueba.
    3. Entrena un modelo de red neuronal simple (PyTorch).
    4. Evalúa y visualiza los resultados.
    """
    path = "Scripts/Rama1/EjericiosPrácticos/5.TodoJunto/Data/2.DataSuperMarket.csv"
    df = pd.read_csv('2.DataSuperMarket.csv')

    df['reward_points'] = np.log(df['reward_points']+1)

    categoryCols = {
        'branch':'one',
        'city':'cat',
        'customer_type':'one',
        'gender':'one',
        'product_name':'cat',
        'product_category':'cat'
    }

    print(df['customer_type'].value_counts())
    df = normalizacion(df, categoryCols)
    print(df.dtypes)
    del df['sale_id']

    analisisDeDatos(df)

    df = estandarizacion(df, ['unit_price', 'quantity', 'tax', 'total_price', 'reward_points'])
    print(df)

    analisisDeDatos(df)

    t = torch.tensor(df.values, dtype=torch.float)
    print(t)

    x_train, x_test, y_train, y_test = aleatorizacion_de_datos(t, [12], 0.8)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    modelo = SimpleNeuralNetwork()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(modelo.parameters(), lr=0.01)

    epochs = 500
    error = []
    for epoch in tqdm(range(epochs)):
        predictions = modelo.forward(x_train)
        loss = criterion(predictions, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 25 == 0:
            print(f'Epoch: {epoch} Loss: {loss.item()}')
            error.append(loss.item())
            
    preds_test = modelo.forward(x_test)
    loss1 = criterion(preds_test, y_test)

    # -------------------
    # Gráficas de resultados
    # -------------------
    x = np.linspace(0, y_test.shape[0], y_test.shape[0])
    ax1[0].scatter(x, y_test, label='Real', color='r')
    ax1[0].scatter(x, preds_test.detach().numpy(), label='Predicción', color='b') 
    ax1[0].legend(title='Leyenda:')
    ax1[0].set_title(f'Predicción vs Real, \nError en testing: {loss1*100:.2f}%', fontsize=20, fontweight='bold') 
    ax1[0].set_xlabel('n Test', fontsize=15)
    ax1[0].set_ylabel('Precio estandarizado', fontsize=15)
       
    ax1[1].plot([num for num in range(int(epochs/25))], error)
    ax1[1].set_title(f'Error en entrenamiento: \n{error[-1]*100:.2f}%', fontsize=20, fontweight='bold')
    ax1[1].set_xlabel('Epochs', fontsize=15)
    ax1[1].set_ylabel('Error', fontsize=15)

    plt.show()
    
    

if __name__ == '__main__':
    main()
