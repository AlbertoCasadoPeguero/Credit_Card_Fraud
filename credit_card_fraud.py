#Importando las librerias
import numpy as np
import pandas as pd
import tensorflow as tf

#Importando el dataset
dataset = pd.read_csv('credit_card.csv')

#Analizando un poco el dataset
dataset.isnull().sum()
dataset.info()
estadisticas = dataset.describe()

#Haciendo un poco de analysis
import seaborn as sns

#La data esta totalmente desbalanceada
sns.countplot('Class',data = dataset)

#Verificando la distribucion
sns.distplot(dataset['Time'])
sns.distplot(dataset['Amount'])

#Analizando los montos de acuerdo a su clase
fraudes = dataset[dataset['Class'] == 1]['Amount']
no_fraudes = dataset[dataset['Class'] == 0]['Amount']
sns.distplot(fraudes)
sns.distplot(no_fraudes)

sns.scatterplot(range(0,len(fraudes)),fraudes)
sns.scatterplot(range(0,len(no_fraudes)),no_fraudes)

#Escalar la data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

transformer = ColumnTransformer(transformers = [('scalador',StandardScaler(),['Time','Amount'])],
                                remainder = 'passthrough')
dataset = transformer.fit_transform(dataset)

#TODO - split the data