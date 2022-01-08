# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 18:54:26 2022

Caso de estudio: Se tiene los datos de un banco, en este se detallan las variables de los clientes y 
se tiene cuáles clientes han abnadonado el bamco, Se desea saber cuál es el procesa que está pasando, por qué 
se están marchando

1 = el clientes se va del banco
0 = el cliente se va del banco

conda install -c conda-forge keras   --> Código para instalar Keras en conda prompt

@author: Jhon Romero
"""


# 1) Preprocedado de datos

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

#Leer Data
ruta_archivo = r"C:\Users\Jhon Romero\Documents\Code++\Curso Deep Learning\PrimeroProblema-RNA\Churn_Modelling.csv"
data = pd.read_csv(ruta_archivo)

#crear matriz de características y el vector de la variable dependiente
    #selecciono desde qué columna voy a tomar mi dataframe x ya que variables como el customerid o Surname no son relevantes
x = data.iloc[:, 3:13].values
y = data.iloc[:,13].values

#Crear variables dummies para las variables categóricas
    #tranformar la columna 1 la cual contiene los paises
label_encoder_x1 = LabelEncoder()
    #Coge los valores de la columna 1 y los ajusta y transforma
x[:,1] = label_encoder_x1.fit_transform(x[:,1])

    #tranformar la columna 2 la cual contiene el género
label_encode_x2 = LabelEncoder()
x[:,2] = label_encode_x2.fit_transform(x[:,2])

    #convertir en varias columnas estas variables dummy
onehotencoder = ColumnTransformer(
    [("one_hot_encoder",OneHotEncoder(categories="auto"),[1])],
    remainder="passthrough"
    )

x = onehotencoder.fit_transform(x)

#para evitar la multicolinealidad se elimina una de las variables dummy ya que si por ejemplo en este caso, 
#en ambos renglos quedan ceros entonces se sabe que el 1 es para la variable que ya no está
x = x[:,1:] #para la columna genero no es necesario lo anterior ya que si es cero se entiende que es el otro valor entonces


#Dividir el data set en conjunto de entranmiento y de testing
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


#Hacer escalado de variables
    #esto se hace porque hay variables que van de 0 a 1, otras van de 0 a 100 y otras tienen escala mayores; se 
    #debe hacer el escalado para que ninguna domine sobre el resto
    #El siguiente código cogerá todas las variables y les dará valores cercanos a 0
"""
Respuesta a pregunta hecha en udemy del por qué en el siguiente código se usa transform() para x_test y 
fit_transform() para x_train

Porque cuando vas a estandarizar el train, no dispones ni de media ni de desviación típica, 
los tienes que calcular en base a los datos, pero cuando vas a estandarizar el conjunto de test, 
ya dispones de los valores usados para estandarizar el train, y para que se estandaricen del mismo modo
 ambos conjuntos (train y test), necesitas utilizar esa misma información.
 
y según el vídeo 28 en el minuto 22:25
"para el x_train se hace el fit transform para calcular el cambio de escala y aplicarlo respectivamente
a la matriz x_train, para el x_test ya se aplica el fit de la linea anterior por ende sólo se hace el transform "
"""

sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)



# 2) Construir la red neuronal articial 

#importar Keras y librerías adicionales

import keras
#inicializaer los parámetros neuronal
from keras.models import Sequential
#declarar y crear capas intermedias de las red 
from keras.layers import Dense
#para matriz de confusión
from sklearn.metrics import confusion_matrix

# Iniciar la red neuronal artificial (RNA)

#crear objeto que será la futura RN

classifier = Sequential()

#Añadir capas de entrada y primer capa oculta
    #añadir capa; Dense es el número de nodos (conexión entre capas) normalmente el units se suele usar la media entre los parámetros de
    #entrada y de salida. Entrada =cantidad de columnas. Salida = Valores que toma la variable dependiente para el caso (si y no, osea 2)
    # 11 columnas de entrada, 1 una de salida = 6 => media entre 11 y 1.... units = 6 =>nodos de salida
    # kernel_initializer  = para inicializar uniformemente los peso iniciales
    # activation = función de activcación inicial relu = rectificado linear unitario
    # input_dim = nodos de salida, suele usarse la cantidad de columnas
    
    #la relación entre input_dim y units es que la RN tomará 11 datos de entrada y los transformará en 6
    
classifier.add(Dense(units = 6,kernel_initializer = "uniform", activation = "relu", input_dim = 11))

#Añadir segunda capa oculta
    #como la primera capa devuelve 6 nodos, ya esta segunda capa sabe que recibe 6 por ende no es necesario el input_dim
    #para el parametro de salida sigo usando sigo usando el 6
classifier.add(Dense(units = 6,kernel_initializer = "uniform", activation = "relu"))

#añadir capa de salida
    #como el resultado es si o no entonces sólo va a haber un nodo en la capa de salida units = 1
    # activation como en la capa de salida quieo que sea una probabilidad (de que el cliente se vaya el banco), 
    # por eso uso la función sigmoide. si hubieran por ejemplo 3 categorías en la capa de salida units, no había sólo una unidad si no 3
    # tanpoco se usaría la función sigmoide. se usaría un rectificador lineal unitario o escalón
classifier.add(Dense(units = 1,kernel_initializer = "uniform", activation = "sigmoid"))


##Compilar la red neuronal
    #optimizer = algoritmo que se usa para encontrar el conjunto optimo de soluciones, o sea gradiente descendiente, estocástico o métod de adam (uno muy usado)
    #loss = función de pérdida para optimizar en este caso la minimizacción de mínimos cuadrados ya que en este caso es "binario" los datos y no categoricos
    #metrics = lista de métricas que quiero que evalúe el modelo, en este caso, aumentar la precisión entre cada medición
classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])


#ajustar la RNA al conjunto de entrenamiento 
    # batch_size = número de bloques en el cual se cambian los parámetros batch_sizesi = 10 => procesar 10 elementos y luego corregir los pesos
    #epochs = cantidad de iteraciones sobre el conjunto de datos. un exceso ni pocas es bueno
classifier.fit(x_train,y_train, batch_size = 10, epochs = 100)



#3) Evalular el modelo y calcular predicciones finales

#variable de predicción
y_predict = classifier.predict(x_test)
#se define una probabilidad por la cual el cliente va a abandonar, supongamos que al banco le interesa los clientes con mayor a 50% de probabilidad
y_predict = (y_predict>0.5)

#crear matriz de confusión: para crear un umbral de categoría para caracterizar un cliente como abandono
cm = confusion_matrix(y_test,y_predict)

#(1523+171)/2000 = 0.847 >si es valor semejante a cuando se entrenó la RN















