from libimports import *

## Carrega um dataset a partir de um path
def load_data(path):
    try:
        with open(path, 'r') as f:
            print("Arquivo ", path, " localizado!")
    except FileNotFoundError as e:
        print(e)
    ## Carrega dados para um dataframe pandas
    temp_df = pd.read_csv(path)
    return temp_df

def split_pipeline(dataframe, split_size = 0.3, categories_qtd = 2, normalize = True, TARGET = 'classe'):
    
    temp_df = dataframe.copy()
    #train_df, test_df = train_test_split(temp_df, test_size = split_size)
    #train_df, val_df = train_test_split(train_df, test_size = split_size/(1-split_size))
    val_df = dataframe[0]
    test_df = dataframe[1]
    train_df = dataframe[2]
    train_labels = np.array(train_df.pop(TARGET))
    val_labels = np.array(val_df.pop(TARGET))
    test_labels = np.array(test_df.pop(TARGET))

    train_labels = tf.keras.utils.to_categorical(train_labels, categories_qtd)
    val_labels = tf.keras.utils.to_categorical(val_labels, categories_qtd)
    test_labels = tf.keras.utils.to_categorical(test_labels, categories_qtd)

    if normalize :
        # mean_norm  = lambda df_input: df_input.apply(lambda x: (x-x.mean())/ x.std(), axis=0)   

        # train_df = mean_norm(train_df)
        # val_df = mean_norm(val_df)
        # test_df = mean_norm(test_df)

        scaler = StandardScaler()
        print(train_df)
        train_df = scaler.fit_transform(train_df)
        val_df = scaler.fit_transform(val_df)
        test_df = scaler.fit_transform(test_df)


    return train_df, train_labels, test_df, test_labels, val_df, val_labels

## Função para criar um modelo keras de rede neural
def make_model2(units, activation, dropout, lr):
    METRICS = [
      keras.metrics.SensitivityAtSpecificity(name='Sen', specificity= 0.5),
      keras.metrics.SpecificityAtSensitivity(name='Spe', sensitivity = 0.5),
      keras.metrics.BinaryAccuracy(name='Acc'),
      keras.metrics.AUC(name='AUC')
    ]

    model = keras.Sequential()
    model.add(tf.keras.Input(shape=(15,)))
    model.add(layers.Dense(units=units, activation=activation))
    if dropout:
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model

def build_model(hp):
    units = hp.Int("units", min_value=32, max_value=512, step=32)
    activation = hp.Choice("activation", ["relu", "tanh"])
    dropout = hp.Boolean("dropout")
    lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    # call existing model-building code with the hyperparameter values.
    model = make_model2(
        units=units, activation=activation, dropout=dropout, lr=lr
    )
    return model
def make_model(output_bias=None, input_shape = 1, output_layers = 1):
    METRICS = [
      keras.metrics.SensitivityAtSpecificity(name='Sen', specificity= 0.5),
      keras.metrics.SpecificityAtSensitivity(name='Spe', sensitivity = 0.5),
      keras.metrics.BinaryAccuracy(name='Acc'),
      keras.metrics.AUC(name='AUC')
    ]

    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    
    model = keras.Sequential([
          keras.layers.Dense( input_shape, activation='relu', input_shape=(input_shape,)),
          keras.layers.Dense( 32, activation='relu'),
          keras.layers.Dropout(0.2),
          keras.layers.Dense( 32, activation='relu'),
          keras.layers.Dropout(0.2),
          #keras.layers.Dense( 16, activation='relu'),
          #keras.layers.Dropout(0.2),
          #keras.layers.Dense( 16, activation='relu', input_shape=(32,)), keras.layers.Dropout(0.1),
          #keras.layers.Dense( 4, activation='relu', input_shape=(16,)), keras.layers.Dropout(0.1),
          keras.layers.Dense( 1 , activation='sigmoid', bias_initializer=output_bias),
    ])

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=1e-4), #15e-4
        loss = keras.losses.BinaryCrossentropy(),
        metrics = METRICS
    )

    return model

def SaveResult(path,resultado,classe):
    f = open('resultados.txt','a')
    f.write('\n')
    f.write(str(classe))
    f.write('\n')
    f.write(path)
    f.write('\n')
    f.write(str(resultado))
    f.write("loss,sen,spe,acc,acu,")
    f.write("\n \n################################# \n")
    f.close()

def gencsv(resultados):
    f = open('resultados.csv','a')
    for i in range(5):
        f.write(str(resultados[i])+',')
    f.write("\n")
    f.close