#Importando las librerias
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

#Importando el dataset
dataset = pd.read_csv('credit_card.csv')

#Analizando un poco el dataset
dataset.isnull().sum()
dataset.info()
estadisticas = dataset.describe()

#La data esta totalmente desbalanceada
sns.countplot('Class',data = dataset)

#Verificando la distribucion
sns.distplot(dataset['Time'])
sns.distplot(dataset['Amount'])

dataset['Amount'].min()
dataset['Amount'].max()

#Analizando los montos de acuerdo a su clase
fraudes = dataset[dataset['Class'] == 1]['Amount']
no_fraudes = dataset[dataset['Class'] == 0]['Amount']
sns.distplot(fraudes)
sns.distplot(no_fraudes)

sns.scatterplot(range(0,len(fraudes)),fraudes)
sns.scatterplot(range(0,len(no_fraudes)),no_fraudes)