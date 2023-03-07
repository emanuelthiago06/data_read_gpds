## Manipulação de dados
import pandas as pd
import numpy as np

## Bibliotecas utilitárias para preparar pipelines de dados
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import keras.utils

# Pra normalizar os dados
from sklearn.preprocessing import StandardScaler

## Bibliotecas de aprendizado de máquina
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner
## Bibliotecas e métricas
from sklearn.metrics import confusion_matrix,roc_curve, auc, accuracy_score, recall_score, precision_score

## Bibliotecas de apresentação
import matplotlib.pyplot as plt
import seaborn as sns ## Trocar por matplotlib
import plotly.express as px

## Bibliotecas úteis
import itertools
from itertools import cycle
from scipy import interp

## Explainer
#import shap
from keras.wrappers.scikit_learn import KerasClassifier
