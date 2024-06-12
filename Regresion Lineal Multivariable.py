import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Cargamos los datos
data = pd.read_csv("C:\\Users\\ivanh\\Desktop\\Escuela\\Octavo Semestre\\Sem.IA2\\Act.03\\dataset_RegresionLinealMultivariable.csv")

#Separamos los datos en X y Y
X = data
X = X.drop(columns=["y"])
Y = data
Y = Y.drop(columns=["x1", "x2"])

# Calculamos la media y la desviación estándar de cada característica en X
means = X.mean()
stds = X.std()

# Normalizamos las características de X
X_normalized = (X - means) / stds
# Asignamos los datos normalizados a X
X = X_normalized

m, n = np.shape(X)
#Agregamos columna de unos a la matriz X.
X.insert(0,"x0", np.ones((m,1)))

#Inicializamos el vector de parametros a
a = np.zeros((n+1,1))

#Inicializar parametros
beta = 0.8
iterMax = 600

#Crear los vectores J y h
J = np.zeros((iterMax,1))
h = np.zeros((m,1))

#Entrenamiento 
for iter in range(iterMax):
    for i in range(m):
        h[i] = np.dot(a.transpose(), X.iloc[i, :])
    J[iter] = np.sum(np.power((h-Y),2), axis=0) / (2*m)
    for j in range(n+1):
        xj = np.mat(X[X.columns[j]])
        xj = xj.transpose()
        a[j] = a[j] - beta*(1/m)*np.sum((h-Y) * xj)

#Dibujar Grafica de convergencia
plt.figure(1)
plt.plot(J)   
plt.ylabel("J")
plt.xlabel("Iteraciones")
plt.title("Gráfica de convergencia")
#Imprimir los resultados de los parametros a y el error J
print( "\n\n", "J=", J[iterMax-1], "a0=", a[0], "a1=", a[1], "a2=", a[2], "\n")

#Predicciones
datosPrueba = [[1, 3000, 4], [1, 1985, 4], [1, 1534, 3]]
costoReal = [539900, 299900, 314900]

for d in range(0, 3):
    casa = datosPrueba[d]
    # Normalizar los valores en la posición 1 y 2 de casa
    casa_normalizado = [casa[0]] + [(casa[i] - means.iloc[i-1]) / stds.iloc[i-1] for i in range(1, len(casa))]
    costo =  h[i] = np.dot(a.transpose(), casa_normalizado) 
    print("Dato de prueba", d+1, ":", " x1=", casa[1], "   x2=", casa[2], "   Salida correcta y=", costoReal[d], "   Predicción h=", costo, "\n")


# Crear una figura 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Agregar datos de entrenamiento
ax.scatter(X_normalized["x1"], X_normalized["x2"], Y, c='red', label='Datos de entrenamiento')

# Agregar datos de prueba
for casa, costo in zip(datosPrueba, costoReal):
    casa_normalizado = [casa[0]] + [(casa[i] - means.iloc[i-1]) / stds.iloc[i-1] for i in range(1, len(casa))]
    ax.scatter(casa_normalizado[1], casa_normalizado[2], costo, c='blue', label='Datos de prueba')

ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y')
ax.set_title('Gráfica de dispersión en 3D')
plt.legend()
plt.show()   