
from http.client import RESET_CONTENT
import multiprocessing
import os
import time
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd

from src.signal.Read import Read
from src.filter.Filter import Filter
from src.processing.Convert import Convert
from src.processing.SignalProcess import SignalProcess
from src.display.Display import Display
import matplotlib.pyplot as plt 

CYCLE_COUNT = 0
FIGSIZE = (2.24, 2.24)
PLOT_LIMS = [-1, 2, -1, 2]  # 0.15 antes
BIDIMENSIONAL_RPS_RESULTS_DIR = 'results/bidimensional_vcg_rps/'

HOME = HOME = os.path.expanduser('~')
DATABASE_PATH = HOME + '/Documentos/ptb_vcg' + '/Databases/PTB-XL/'
print(DATABASE_PATH)
count_signals = 0
count_bad_signals = 0
count_error = 0

def cycleProcess(cycleNumber,errors) -> None:
    try:

            # PRO CASO DE USAR APENAS UMA PARTE DE UM CICLO
            # inferior_slice_lim = 50
            # superior_slice_lim = 300

        signalOfCycle = vcg.getSignalCycle(
            cycleNumber)  # [inferior_slice_lim:superior_slice_lim]

            # PRO CASO DE USAR APENAS UMA PARTE DE UM CICLO
            # if signalOfCycle.shape[0] != superior_slice_lim - inferior_slice_lim:
            #     return
        variables = {
                0: "x",
                1: "y",
                2: "z"
            }
        limit = 450
        if signalOfCycle.shape[0] <= limit:
            limit = signalOfCycle.shape[0]
        for vcg_lead in range(vcg.signal.shape[1]):
            os.makedirs(f'/home/gpds/Documentos/ptb_vcg/dissertacao-main/Results_S-ST/{dataset}/{currentSignalLabel}/{currentSignalName}/{cycleNumber}', exist_ok = True)
        for tau in [1,2]:
            vcgLeadsOfCycleRPS = np.zeros(
            (limit-50 - n_samples_to_delay*tau, 2))
            for vcg_lead in range(vcg.signal.shape[1]):
                vcgLeadsOfCycleRPS[:, :] = SignalProcess.reconstructPhaseSpace(signalOfCycle[50:limit, vcg_lead], n_samples_to_delay*tau)
                Display(vcgLeadsOfCycleRPS, FIGSIZE, PLOT_LIMS).save_RPS_image(f'/home/gpds/Documentos/ptb_vcg/dissertacao-main/Results_S-ST/{dataset}/{currentSignalLabel}/{currentSignalName}/{cycleNumber}/RPS_{variables[vcg_lead]}_tau00{tau}s.png', 'off')
    except:
        errors+=1

datasets = ['training', 'val', 'test']

start = time.time()
signal_x_list = []
signal_y_list = []
signal_z_list = []
cont_cycles = 1


for dataset in datasets:

    BIDIMENSIONAL_RPS_DATASET_RESULTS_DIR = f'{BIDIMENSIONAL_RPS_RESULTS_DIR}{dataset}/'

    for label in ['0', '1']:
        Path(f'{BIDIMENSIONAL_RPS_DATASET_RESULTS_DIR}{label}/') .mkdir(
            parents=True, exist_ok=True)

    signal_list = pd.read_csv(f'/home/gpds/Documentos/ptb_vcg/dissertacao-main/Results/{dataset}_filenames.csv', index_col=0)

    # Contador para teste
    count = 0
    number = 0
    for numCurrentSignal in range(0, signal_list.shape[0]):
        currentSignal = signal_list.iloc[numCurrentSignal, :]

        currentSignalPath = ''.join(
            [DATABASE_PATH, currentSignal['filename_hr'], '.dat'])
        

        currentSignalName = currentSignalPath.split('/')[-1].split('.')[0]
        currentSignalLabel = currentSignal['MI']
        ecg = Read.read_ecg_dat(currentSignalPath)
        ecg.signal = Filter.fir_biosppy(ecg.signal, ecg.samplingFreq)
        vcg = Convert.ecg_to_vcg(ecg)      
        time_delay_seconds = 0.01
        n_samples_to_delay = int(time_delay_seconds*vcg.samplingFreq)

        numberOfCycles = vcg.rPeakIndexes.shape[0]-1
        for cycle in range(1, numberOfCycles+1):
            cycleProcess(cycle,count_error)
         #Incrementa o contador para teste
            count += 1
        print(f'{dataset}: Signal {numCurrentSignal+1}/{signal_list.shape[0]}')
        cont_cycles = 0
    # Finaliza caso o contador para teste chegue em 100
    #     if(count == 100):
    #         break
    # break
print(f'\n\nElapsed time: {(time.time() - start)} seconds')

print(f'\nDos {count_signals} sinais, {count_bad_signals} eram menor que 450, e {count_error} sinais com erro de execução')
