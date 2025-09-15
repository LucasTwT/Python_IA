import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm 
import torch.nn as nn
import torch.nn.functional as F

fig2, ax1 = plt.subplots(1,2, figsize=(7,3))

#Arquitecura de la red neuronal:

class SimpleNeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features=13, out_features=8)
        self.fc2 = nn.Linear(in_features=8, out_features=1)
    def forward(self, x: torch.tensor) -> torch.tensor:
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def analisisDeDatos(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(2,2, figsize=(7,3))
    columns = ['unit_price', 'total_price', 'quantity', 'reward_points']
    for i in tqdm(range(len(columns))):
        if i > 1:
            ax[1, i-2].hist(df[columns[i]], bins=40, edgecolor='k', label=f'{columns[i]} MODE{df[columns[i]].mode()}')
            ax[1, i-2].legend(title='Leyenda:')
        else:
            ax[0, i].hist(df[columns[i]], bins=40, edgecolor='k', label=f'{columns[i]} MODE{df[columns[i]].mode()}')
            ax[0,i].legend(title='Leyenda:')        
 

def normalizacion (df: pd.DataFrame, colsINDEX: dict) -> pd.DataFrame:
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
    new_df = df[columns]
    new_df = (new_df - new_df.mean()) / new_df.std() 
    df.drop(columns=columns, inplace=True)
    df = pd.concat([df, new_df], axis=1).astype(float)
    return df  

def aleatorizacion_de_datos(t: torch.tensor, columns_target: list, percents: float) -> tuple:
    index = torch.randperm(t.shape[0])
    t = t[index]
    x = t[:, [i for i in tqdm(range(t.shape[1])) if i not in columns_target]]
    y = t[:, [i for i in tqdm(range(t.shape[1])) if i in columns_target]]
    x_train, x_test = torch.split(x,int(x.shape[0]*percents), dim=0)
    y_train, y_test  = torch.split(y,int(y.shape[0]*percents), dim=0)
    return x_train, x_test, y_train, y_test

def main() -> None:
    path = "Scripts/Rama1/EjericiosPrácticos/5.TodoJunto/Data/2.DataSuperMarket.csv"
    df = pd.read_csv('2.DataSuperMarket.csv')
    df['reward_points'] = np.log(df['reward_points']+1)
    categoryCols = {'branch':'one','city':'cat','customer_type':'one','gender':'one','product_name':'cat','product_category':'cat'}
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