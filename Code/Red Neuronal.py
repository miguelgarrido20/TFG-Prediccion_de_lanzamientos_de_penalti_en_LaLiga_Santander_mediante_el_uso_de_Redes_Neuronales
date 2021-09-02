import pandas as pd
import numpy as np
import sklearn
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.backend import dtype
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Conv1D, Flatten, Activation, Input
from keras.layers.core import Dropout, Reshape
from keras.optimizers import Adam
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from mlxtend.plotting import plot_decision_regions

import warnings
warnings.filterwarnings("ignore")

def matriz_confusion(y_test, y_pred, naming):

    # de cada salida se desea su matriz de confusión con la cantidad de valores y su valor normalizado
    cm = confusion_matrix(y_test, y_pred, normalize=None)
    cm_normalize = confusion_matrix(y_test, y_pred, normalize='true')

    # scored
    if naming == 0:
        labels = ['No Gol', 'Gol']
        
        fig, ax = plt.subplots(figsize=(10,5))
        ax.matshow(cm)
        plt.title('Matriz de confusion', fontsize=20)
        plt.ylabel('Valor real', fontsize=15)
        plt.xlabel('Valor predicho', fontsize=15)
        ax.set_yticklabels([''] + labels, fontsize="10")
        ax.set_xticklabels([''] + labels, fontsize="10" )
        
        for(i,j),z in np.ndenumerate(cm):
            ax.text(j,i, format(z), ha='center', va='center')
        plt.show()
        
        fig, ax = plt.subplots(figsize=(10,5))
        ax.matshow(cm_normalize)
        plt.title('Matriz de confusion normalizada', fontsize=20)
        plt.ylabel('Valor real', fontsize=15)
        plt.xlabel('Valor predicho', fontsize=15)
        ax.set_yticklabels([''] + labels, fontsize="10")
        ax.set_xticklabels([''] + labels, fontsize="10" )
        
        for(i,j),z in np.ndenumerate(cm_normalize):
            ax.text(j,i, '{:0.2f}'.format(z), ha='center', va='center')
        plt.show()

    # saved
    elif naming == 1:
        labels = ['No Parada', 'Parada']

        fig, ax = plt.subplots(figsize=(10,5))
        ax.matshow(cm)
        plt.title('Matriz de confusion', fontsize=20)
        plt.ylabel('Valor real', fontsize=15)
        plt.xlabel('Valor predicho', fontsize=15)
        ax.set_yticklabels([''] + labels, fontsize="10")
        ax.set_xticklabels([''] + labels, fontsize="10" )
        
        for(i,j),z in np.ndenumerate(cm):
            ax.text(j,i, format(z), ha='center', va='center')
        plt.show()

        fig, ax = plt.subplots(figsize=(10,5))
        ax.matshow(cm_normalize)
        plt.title('Matriz de confusion normalizada', fontsize=20)
        plt.ylabel('Valor real', fontsize=15)
        plt.xlabel('Valor predicho', fontsize=15)
        ax.set_yticklabels([''] + labels, fontsize="10")
        ax.set_xticklabels([''] + labels, fontsize="10" )
        
        for(i,j),z in np.ndenumerate(cm_normalize):
            ax.text(j,i, '{:0.2f}'.format(z), ha='center', va='center')
        plt.show()

    # kick_direction y keeper_direction
    else:
        directions = ['LT','CT','RT','LC','CC','RC','LD','CD','RD']

        fig, ax = plt.subplots(figsize=(10,5))
        ax.matshow(cm)
        plt.title('Matriz de confusion', fontsize=20)
        plt.ylabel('Valor real', fontsize=15)
        plt.xlabel('Valor predicho', fontsize=15)
        ax.set_yticklabels([''] + directions, fontsize="10")
        ax.set_xticklabels([''] + directions, fontsize="10" )

        for(i,j),z in np.ndenumerate(cm):
            ax.text(j,i, format(z), ha='center', va='center')
        plt.show()

        fig, ax = plt.subplots(figsize=(10,5))
        ax.matshow(cm_normalize)
        plt.title('Matriz de confusion normalizada', fontsize=20)
        plt.ylabel('Valor real', fontsize=15)
        plt.xlabel('Valor predicho', fontsize=15)
        ax.set_yticklabels([''] + directions, fontsize="10")
        ax.set_xticklabels([''] + directions, fontsize="10" )

        for(i,j),z in np.ndenumerate(cm_normalize):
            ax.text(j,i, '{:0.2f}'.format(z), ha='center', va='center')
        plt.show()

def pca_method1(dataset):
    # PCA: Análisis de los componentes principales (www.pharos.sh)
    '''
    encoder = LabelEncoder()

    # Now apply the transformation to all the columns:
    for col in df.columns:
        df[col] = encoder.fit_transform(df[col])

    X_features = df.iloc[:,:11]
    y_label = df.iloc[:,11:]

    # Scale the features
    scaler = StandardScaler()
    X_features = scaler.fit_transform(X_features)
    '''

    X_features = dataset[:,:11]
    y_label = dataset[:,11:]
    print(X_features)

    # Visualize
    pca = PCA()
    pca.fit_transform(X_features)
    pca_variance = pca.explained_variance_

    # Ver la varianza explicada de cada una de los componentes principales y la varianza acumulada
    print('----------------------------------------------------------')
    print('Varianza de cada componente principal y varianza acumulada')
    print('----------------------------------------------------------')
    var_acum= np.cumsum(pca_variance)
    plt.bar(range(len(pca_variance)), pca_variance)
    plt.plot(range(len(pca_variance)), var_acum)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.bar(range(11), pca_variance, alpha=0.5, align='center', label="individual variance") #range = nº columnas
    plt.legend()
    plt.ylabel('Variance ratio')
    plt.xlabel('Principal components')
    plt.show()

    # diagrama de dispersión de la clasificación de puntos en función de estas 8 características
    pca2 = PCA(n_components=9)
    pca2.fit(X_features)
    x_3d = pca2.transform(X_features)
    print(x_3d)

    plt.figure(figsize=(8,6))
    plt.scatter(x_3d[:,0], x_3d[:,1], c=dataset[:,0]) # c=df['NOMBRECOLUMNA']
    plt.show()
    '''
    # diagrama de dispersión de la clasificación de puntos para las 2 características principales
    pca3 = PCA(n_components=2)
    pca3.fit(X_features)
    x_3d = pca3.transform(X_features)

    plt.figure(figsize=(8,6))
    plt.scatter(x_3d[:,0], x_3d[:,1], c=dataset[:,0])# c=df['NOMBRECOLUMNA']
    plt.show()'''
    
    # Recostruccion de las proyecciones
    # ==============================================================================
    from sklearn.pipeline import make_pipeline

    df = pd.DataFrame(dataset[:,:x_3d.shape[1]])

    pca_pipe = make_pipeline(StandardScaler(), PCA())
    pca_pipe.fit(df)
    
    # Proyección de las observaciones de entrenamiento
    # ==============================================================================
    proyecciones = pca_pipe.transform(X=df)
    proyecciones = pd.DataFrame(
        proyecciones,
        index   = df.index
    )
    print(proyecciones.head())

    recostruccion = pca_pipe.inverse_transform(X=proyecciones)
    recostruccion = pd.DataFrame(
                        recostruccion,
                        index   = df.index
    )
    print('------------------')
    print('Valores originales')
    print('------------------')
    print(recostruccion.head())

    return recostruccion.values 

def pca_method2(df):
    # https://www.cienciadedatos.net/documentos/py19-pca-python.html
    # Tratamiento de datos
    # ==============================================================================
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    # Gráficos
    # ==============================================================================
    import matplotlib.pyplot as plt
    import matplotlib.font_manager
    from matplotlib import style
    style.use('ggplot') or plt.style.use('ggplot')

    # Preprocesado y modelado
    # ==============================================================================
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import scale

    # Configuración warnings
    # ==============================================================================
    import warnings
    warnings.filterwarnings('ignore')

    print(df)

    print('----------------------')
    print('Media de cada variable')
    print('----------------------')
    print(df.mean(axis=0))

    print('-------------------------')
    print('Varianza de cada variable')
    print('-------------------------')
    print(df.var(axis=0))

    x = df.iloc[:,0:11]
    y = df.iloc[:,11:]
    df = x

    # Entrenamiento modelo PCA con escalado de los datos
    # ==============================================================================
    pca_pipe = make_pipeline(StandardScaler(), PCA())
    pca_pipe.fit(df)

    # Se extrae el modelo entrenado del pipeline
    modelo_pca = pca_pipe.named_steps['pca']

    # Se combierte el array a dataframe para añadir nombres a los ejes.
    print(pd.DataFrame(
        data    = modelo_pca.components_,
        columns = df.columns,
        index   = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11']
        #index   = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11', 'PC12', 'PC13', 'PC14', 'PC15']

    ))

    # Heatmap componentes
    # ==============================================================================
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 2))
    componentes = modelo_pca.components_
    plt.imshow(componentes.T, cmap='viridis', aspect='auto')
    plt.yticks(range(len(df.columns)), df.columns)
    plt.xticks(range(len(df.columns)), np.arange(modelo_pca.n_components_) + 1)
    plt.grid(False)
    plt.colorbar()
    plt.show()

    # Porcentaje de varianza explicada por cada componente
    # ==============================================================================
    print('----------------------------------------------------')
    print('Porcentaje de varianza explicada por cada componente')
    print('----------------------------------------------------')
    print(modelo_pca.explained_variance_ratio_)

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
    ax.bar(
        x = np.arange(modelo_pca.n_components_) + 1,
        height = modelo_pca.explained_variance_ratio_
    )

    for x, y in zip(np.arange(len(df.columns)) + 1, modelo_pca.explained_variance_ratio_):
        label = round(y, 2)
        ax.annotate(
            label,
            (x,y),
            textcoords="offset points",
            xytext=(0,10),
            ha='center'
        )

    ax.set_xticks(np.arange(modelo_pca.n_components_) + 1)
    ax.set_ylim(0, 1.1)
    ax.set_title('Porcentaje de varianza explicada por cada componente')
    ax.set_xlabel('Componente principal')
    ax.set_ylabel('Por. varianza explicada')
    plt.show()

    # Proyección de las observaciones de entrenamiento
    # ==============================================================================
    proyecciones = pca_pipe.transform(X=df)
    proyecciones = pd.DataFrame(
        proyecciones,
        columns = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', 'PC11'],
        index   = df.index
    )
    print(proyecciones.head())

    # Recostruccion de las proyecciones
    # ==============================================================================
    recostruccion = pca_pipe.inverse_transform(X=proyecciones)
    recostruccion = pd.DataFrame(
                        recostruccion,
                        columns = df.columns,
                        index   = df.index
    )
    print('------------------')
    print('Valores originales')
    print('------------------')
    print(recostruccion.head())

    print('---------------------')
    print('Valores reconstruidos')
    print('---------------------')
    print(df.head())

    return recostruccion

def lda(dataset):
    
    x_test = dataset[0:1047,0:11]
    y_test = dataset[0:1047,11:]
    x_train = dataset[1048:,0:11]
    y_train = dataset[1048:,11:]
    
    # columnas 11 y 12
    features = dataset[:,:11]
    labels = dataset[:,11:]
    print(labels)

    # esto para columna 13 y 14 que son de 9 clases (las direcciones)
    bins_original = labels[:,2:]
    bins = np.linspace(labels.min(), labels.max(), 9)
    bins_original = np.digitize(labels, bins)
    print(bins_original)
    labels = labels[:,0:1]

    x_test = dataset[0:1047,0:11]
    y_test = labels[0:1047,:] #labels para col 3 y 4
    x_train = dataset[1048:,0:11]
    y_train = labels[1048:,:] #labels para col 3 y 4

    model = LDA()
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted') #columna 1 y 2

    print("Accuracy: {}".format(acc))
    print("F1 Score: {}".format(f1))

    logreg_clf = LogisticRegression()
    logreg_clf.fit(x_train, y_train)
    preds = logreg_clf.predict(x_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted') #columna 1 y 2

    print("Accuracy: {}".format(acc))
    print("F1 Score: {}".format(f1))

    LDA_transform = LDA(n_components=1)
    LDA_transform.fit(features, labels)
    features_new = LDA_transform.transform(features)

    # Print the number of features
    print('Original feature #:', features.shape[1])
    print('Reduced feature #:', features_new.shape[1])

    # Print the ratio of explained variance
    print('Ratio of explained variance :', LDA_transform.explained_variance_ratio_)

    return features_new

archivo = 'C:/Users/Miguel/Desktop/TFG/Code/LaLiga_data_augmentation_gaussian_noise.xlsx'

df = pd.read_excel(archivo)

# Convertimos el dataset de una variable DataFrame a una array numpy.ndarray, más fácil de operar
dataset = df.values

# se define las variables que tendrán los valores string a los que corresponde el número en 'dataset'
season, date, player, goalkeeper, team_player, team_goalkeeper, match, final_result = [], [], [], [], [], [], [], []
# match_week, time_penalty son integer ya
# foot --> R = 0, L = 1
# direction de 0 a 8

for j in range(len(dataset[0])):
    for i in range(len(dataset)):
        # season
        if j == 0:
            if dataset[i][j] in season:
                dataset[i][j] = season.index(dataset[i][j])
            else: 
                season.append(dataset[i][j])
                dataset[i][j] = season.index(dataset[i][j])
        
        # date
        elif j == 2:
            if dataset[i][j] in date:
                dataset[i][j] = date.index(dataset[i][j])
            else: 
                date.append(dataset[i][j])
                dataset[i][j] = date.index(dataset[i][j])
        
        # player
        elif j == 3:
            if dataset[i][j] in player:
                dataset[i][j] = player.index(dataset[i][j])
            else: 
                player.append(dataset[i][j])
                dataset[i][j] = player.index(dataset[i][j])
        
        # goalkeeper
        elif j == 4:
            if dataset[i][j] in goalkeeper:
                dataset[i][j] = goalkeeper.index(dataset[i][j])
            else: 
                goalkeeper.append(dataset[i][j])
                dataset[i][j] = goalkeeper.index(dataset[i][j])
        
        # team_player
        elif j == 5:
            if dataset[i][j] in team_player:
                dataset[i][j] = team_player.index(dataset[i][j])
            else: 
                team_player.append(dataset[i][j])
                dataset[i][j] = team_player.index(dataset[i][j])
        
        # team_goalkeeper
        elif j == 6:
            if dataset[i][j] in team_goalkeeper:
                dataset[i][j] = team_goalkeeper.index(dataset[i][j])
            else: 
                team_goalkeeper.append(dataset[i][j])
                dataset[i][j] = team_goalkeeper.index(dataset[i][j])
        
        # match
        elif j == 7:
            if dataset[i][j] in match:
                dataset[i][j] = match.index(dataset[i][j])
            else: 
                match.append(dataset[i][j])
                dataset[i][j] = match.index(dataset[i][j])
       
        # final result
        elif j == 9:
            if dataset[i][j] in final_result:
                dataset[i][j] = final_result.index(dataset[i][j])
            else: 
                final_result.append(dataset[i][j])
                dataset[i][j] = final_result.index(dataset[i][j])
        
        # foot
        if dataset[i][j] == 'R':
            dataset[i][j] = 0
        elif dataset[i][j] == 'L':
            dataset[i][j] = 1
        
        # dirección
        if dataset[i,j] == 'LT':
            dataset[i,j] = 0
        elif dataset[i,j] == 'CT':
            dataset[i,j] = 1
        elif dataset[i,j] == 'RT':
            dataset[i,j] = 2
        elif dataset[i,j] == 'LC':
            dataset[i,j] = 3
        elif dataset[i,j] == 'CC':
            dataset[i,j] = 4
        elif dataset[i,j] == 'RC':
            dataset[i,j] = 5
        elif dataset[i,j] == 'LD':
            dataset[i,j] = 6
        elif dataset[i,j] == 'CD':
            dataset[i,j] = 7
        elif dataset[i,j] == 'RD':
           dataset[i,j] = 8

# dataset es una lista, keras no soporta este formato, lo pasamos a Numpy array
dataset = np.asarray(dataset).astype(np.int)

# se normaliza el dataset
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
dataset = min_max_scaler.fit_transform(dataset)

# se divide el dataset en conjunto de entrenamiento y de validación
x_test = dataset[0:1048,0:11]
x_train = dataset[1049:,0:11]
y_test = dataset[0:1048,11:]
y_train = dataset[1049:,11:]

# one hot encoding para las columnas 3 y 4 (kick_direction, keeper_direction)
softmax_train_1 = (y_train[:,2:3]*8).astype(int)
softmax_train_2 = (y_train[:,3:]*8).astype(int)
softmax_test_1 = (y_test[:,2:3]*8).astype(int)
softmax_test_2 = (y_test[:,3:]*8).astype(int)

enc = OneHotEncoder()

onehot_encoder = OneHotEncoder(sparse=False)
softmax_train_1 = softmax_train_1.reshape(len(softmax_train_1), 1)
softmax_train_1 = onehot_encoder.fit_transform(softmax_train_1)

softmax_train_2 = softmax_train_2.reshape(len(softmax_train_2), 1)
softmax_train_2 = onehot_encoder.fit_transform(softmax_train_2)

softmax_test_1 = softmax_test_1.reshape(len(softmax_test_1), 1)
softmax_test_1 = onehot_encoder.fit_transform(softmax_test_1)

softmax_test_2 = softmax_test_2.reshape(len(softmax_test_2), 1)
softmax_test_2 = onehot_encoder.fit_transform(softmax_test_2)

# reducción dimensionalidad
## PCA método 1
#X_features = pca_method1(dataset)

##PCA método 2
#df = pd.DataFrame(dataset)
#X_features = pca_method2(df)
#print(X_features.shape[1])

##LDA
#X_features = lda(dataset)
#print(X_features.shape)
#X_features = min_max_scaler.fit_transform(X_features)
#print(X_features)

#x_test = X_features[0:1048,:]
#x_train = X_features[1049:,:]

# creamos modelo
model = Sequential()

# añadimos las capas al modelo
model.add(Dense(16, activation='relu', input_shape=(x_test.shape[1],))) #(x_test.shape[1],)
model.add(Dense(64, activation='relu'))

# dividimos el modelo en 2 salidas, la de gol/no gol, y la de las direcciones
input_model = Input(shape=(x_test.shape[1],)) 

# extraemos la representación del modelo
features = model(input_model)

# determinamos validity para la salida scored y saved, y label para las salidas kick_direction y keeper_direction
validity_1 = Dense(1, activation='sigmoid')(features)
validity_2 = Dense(1, activation='sigmoid')(features)
label_1 = Dense(9, activation='softmax')(features)
label_2 = Dense(9, activation='softmax')(features)

model_2 = Model(input_model, [validity_1, validity_2, label_1, label_2])

# definimos la pérdida y el optimizador
losses = ['binary_crossentropy','binary_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy']
optimizer = Adam(lr=0.0002)

# compilamos el modelo usando la precisión como medidor del funcionamiento del modelo
model_2.compile(optimizer=optimizer, loss=losses, metrics=['accuracy'])

# entrenamos el modelo
## guardamos el entrenamiento en la variable 'history' para poder hacer plot luego
history = model_2.fit(x_train, [y_train[:,0:1], y_train[:,1:2], softmax_train_1, softmax_train_2], validation_data=(x_test, [y_test[:,0:1], y_test[:,1:2], softmax_test_1, softmax_test_2]), epochs=1000, batch_size=None)

# evaluamos el modelo
print('\nEvaluate: \n')
model_2.evaluate(x_test, [y_test[:,0:1], y_test[:,1:2], softmax_test_1, softmax_test_2], verbose=True)

# comprobamos la configuración del modelo
print('\n')
model.summary()
model_2.summary()

# predecimos los valores del conjunto de validación
preds = model_2.predict(x_test[:])
print('\nPredicción con model_2')
print(preds[0])
print(preds[1])
print(preds[2])
print(preds[3])

# imprimimos los valores reales de las predicciones hechas justo antes
print('\nValor real de la predicción anterior')
print(y_test[:])

# creamos variables donde almacenar la dirección con mayor porcentaje de softmax
preds_1 = np.zeros(shape=(y_test.shape[0],1), dtype=int)
preds_2 = np.zeros(shape=(y_test.shape[0],1), dtype=int)

# convertimos las predicciones de gol/no gol y parada/no parada en 0/1
for i in range(y_test.shape[0]):
    if preds[0][i,0:1] >= 0.5:
         preds[0][i,0:1] = 1 
    else: preds[0][i,0:1] = 0

for i in range(y_test.shape[0]):
    if preds[1][i,0:1] >= 0.5:
         preds[1][i,0:1] = 1 
    else: preds[1][i,0:1] = 0

# de las 9 direcciones de softmax, escogemos la posición con mayor porcentaje que es la dirección predicha
for i in range(y_test.shape[0]):
    preds_1[i] = int(np.where(max(preds[2][i,:]) == preds[2][i,:])[0])
    preds_2[i] = int(np.where(max(preds[3][i,:]) == preds[3][i,:])[0])

for j in range(len(y_test[0])):
    for i in range(len(y_test)):
        if j == 2 or j == 3:
            y_test[i,j] = (y_test[i,j]*8) #*8 para pasar de float normalizado entre 0 y 1 a int entre 0 y 8 (9 direcciones)

# convertimos los valores float a int para poder hacer plot()
y_test = np.array(y_test,dtype=int)
preds[0] = np.array(preds[0],dtype=int)
preds[1] = np.array(preds[1],dtype=int)
preds_1 = np.array(preds_1,dtype=int)
preds_2 = np.array(preds_2,dtype=int)

# matrices de confusión
matriz_confusion(y_test[:,0:1], preds[0], 0)
matriz_confusion(y_test[:,1:2], preds[1], 1)
matriz_confusion(y_test[:,2:3], preds_1, 2)
matriz_confusion(y_test[:,3:4], preds_2, 2)

print('\n')

# F1_SCORE
## scored
acc_1 = accuracy_score(y_test[:,0:1], preds[0])
f1_1 = f1_score(y_test[:,0:1], preds[0])

print("Accuracy columna scored: {}".format(acc_1))
print("F1 Score columna scored: {}".format(f1_1))

## saved
acc_2 = accuracy_score(y_test[:,1:2], preds[1])
f1_2 = f1_score(y_test[:,1:2], preds[1])

print("Accuracy columna saved: {}".format(acc_2))
print("F1 Score columna saved: {}".format(f1_2))

## kick_direction
acc_3 = accuracy_score(y_test[:,2:3], preds_1)
f1_3 = f1_score(y_test[:,2:3], preds_1, average='weighted')

print("Accuracy columna kick_direction: {}".format(acc_3))
print("F1 Score columna kick_direction: {}".format(f1_3))

## keeper_direction
acc_4 = accuracy_score(y_test[:,3:], preds_2)
f1_4 = f1_score(y_test[:,3:], preds_2, average='weighted')

print("Accuracy columna keeper_direction: {}".format(acc_4))
print("F1 Score columna keeper_direction: {}".format(f1_4))

# graficamos los valores de pérdida y precisión para el conjunto de entrenamiento y de validación para ver si nuestro modelo sufre overfitting
## primera capa = sigmoid
plt.plot(history.history['dense_2_accuracy'])
plt.plot(history.history['val_dense_2_accuracy'])

plt.title('model accuracy dense_2')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['dense_2_loss'])
plt.plot(history.history['val_dense_2_loss'])

plt.title('model loss dense_2')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

## segunda capa = sigmoid
plt.plot(history.history['dense_3_accuracy'])
plt.plot(history.history['val_dense_3_accuracy'])

plt.title('model accuracy dense_3')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['dense_3_loss'])
plt.plot(history.history['val_dense_3_loss'])

plt.title('model loss dense_3')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

## tercera capa = softmax
plt.plot(history.history['dense_4_accuracy'])
plt.plot(history.history['val_dense_4_accuracy'])

plt.title('model accuracy dense_4')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['dense_4_loss'])
plt.plot(history.history['val_dense_4_loss'])

plt.title('model loss dense_4')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

## cuarta capa = softmax
plt.plot(history.history['dense_5_accuracy'])
plt.plot(history.history['val_dense_5_accuracy'])

plt.title('model accuracy dense_5')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['dense_5_loss'])
plt.plot(history.history['val_dense_5_loss'])

plt.title('model loss dense_5')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()