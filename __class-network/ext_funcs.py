from libimports import *

INPUT_SIZE = 16

def iniciate_input_size(number):
    INPUT_SIZE = number

def split_pipeline(dataframe, split_size = 0.3, categories_qtd = 2, normalize = True, TARGET = 'classe'):
    
    temp_df = dataframe.copy()
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
        scaler = StandardScaler()
        print(train_df)
        train_df = scaler.fit_transform(train_df)
        val_df = scaler.fit_transform(val_df)
        test_df = scaler.fit_transform(test_df)


    return train_df, train_labels, test_df, test_labels, val_df, val_labels

def make_model(units, activation, dropout, lr, input_size = INPUT_SIZE):
    METRICS = [
      keras.metrics.SensitivityAtSpecificity(name='Sen', specificity= 0.5),
      keras.metrics.SpecificityAtSensitivity(name='Spe', sensitivity = 0.5),
      keras.metrics.BinaryAccuracy(name='Acc'),
      keras.metrics.AUC(name='AUC')
    ]

    model = keras.Sequential()
    model.add(tf.keras.Input(shape=(input_size,)))
    model.add(layers.Dense(units=units, activation=activation))
    if dropout:
        model.add(layers.Dropout(rate=0.25))
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy",keras.metrics.SensitivityAtSpecificity(name='Sen', specificity= 0.5),
                    tf.keras.metrics.Precision(name='Pres'),
                    tf.keras.metrics.Recall(name = "recall")
                ],
    )
    return model

def build_model(hp, input_size = INPUT_SIZE):
    units = hp.Int("units", min_value=32, max_value=512, step=32)
    activation = hp.Choice("activation", ["relu", "tanh"])
    dropout = hp.Boolean("dropout")
    lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
    # call existing model-building code with the hyperparameter values.
    model = make_model(
        units=units, activation=activation, dropout=dropout, lr=lr, input_size=input_size
    )
    return model