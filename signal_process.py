import re
import pandas as pd
import numpy as np
import wfdb
import ast
import os
#from src.GPDSHelper import GPDSHelper as gpds
#from src.ECG import ECG
from src.Metrics import Metrics as met
from timeit import default_timer as timer

start = timer()
total_signal = 1
ROOT_NAME = "Results_S-ST"

regex_signal_name = r"^Results_S-ST/(\w*)/(?P<classe>\d)/(?P<numero>\d*)_hr"
cont =0
contSignal = 1
dir_exe = ['0','1']

loops = ["val","training","test"]
n = 0
for j in loops:
    dados = pd.DataFrame(columns=('signalName','classe','split',
                              'd1_X', 'd2_X', 'd_dif_X', 'sim_X', 'd_pond_X',
                              'd1_Y', 'd2_Y', 'd_dif_Y', 'sim_Y', 'd_pond_Y',
                              'd1_Z', 'd2_Z', 'd_dif_Z', 'sim_Z', 'd_pond_Z'))

    for root,dir,files in os.walk(f"{ROOT_NAME}/{j}",topdown=False):
        try:
            tr = root+'/'+files[1]
            total_signal+=1
        except:
            tr = 0
    print(total_signal)
    for root,dir,files in os.walk(f"{ROOT_NAME}/{j}",topdown=False):
        try:
            temp = files[0]
        except:
            continue    
        match = re.search(regex_signal_name,root)
        signalName = match.group("numero")
        classe = match.group("classe")
        xt1 = root+'/'+files[1]
        yt1 = root+'/'+files[3]
        zt1 = root+'/'+files[4]
        xt2 = root+'/'+files[2]
        yt2 = root+'/'+files[5]
        zt2 = root+'/'+files[0]
             #PARAMÊTROS - DERIVAÇÃO x
        m_fixo_X = met(xt1, xt2, block_size=20)
        cm_fixo1_X = m_fixo_X.count_method(m_fixo_X.image1)
        cm_fixo2_X = m_fixo_X.count_method(m_fixo_X.image2)
        dif_fixo_X = m_fixo_X.dif_method()
        sim_fixo_X = m_fixo_X.sim_method()
        d_pond_fixo_X = m_fixo_X.pond_method()
             #PARAMETROS - DERIVACAO Y
        m_fixo_Y = met(yt1, yt2, block_size=20)
        cm_fixo1_Y = m_fixo_Y.count_method(m_fixo_Y.image1)
        cm_fixo2_Y = m_fixo_Y.count_method(m_fixo_Y.image2)
        dif_fixo_Y = m_fixo_Y.dif_method()
        sim_fixo_Y = m_fixo_Y.sim_method()
        d_pond_fixo_Y = m_fixo_Y.pond_method()
             #PARAMETROS - DERIVACAO Z
        m_fixo_Z = met(zt1, zt2, block_size=20)
        cm_fixo1_Z = m_fixo_Z.count_method(m_fixo_Z.image1)
        cm_fixo2_Z = m_fixo_Z.count_method(m_fixo_Z.image2)
        dif_fixo_Z = m_fixo_Z.dif_method()
        sim_fixo_Z = m_fixo_Z.sim_method()
        d_pond_fixo_Z = m_fixo_Z.pond_method()
        dados.loc[cont] = [signalName, classe, j, cm_fixo1_X, cm_fixo2_X, dif_fixo_X, sim_fixo_X, d_pond_fixo_X,
                            cm_fixo1_Y, cm_fixo2_Y, dif_fixo_Y, sim_fixo_Y, d_pond_fixo_Y,
                            cm_fixo1_Z, cm_fixo2_Z, dif_fixo_Z, sim_fixo_Z, d_pond_fixo_Z]    
        cont = cont + 1
        contSignal+=1
        print(f'Signal processed: {contSignal}/{total_signal}\n')
            # Pra teste apenas para limitar a quantidade de sinais processados
    cont = 0
    contSignal = 0
    total_signal = 0
    try:
        os.makedirs(f'resultados_csv/Results_ST')
    except:
        pass
    dados.to_csv(f'resultados_csv/Results_ST/dados_{j}_ptb.csv', mode='w')
    del dados
# Finalização do 
print(n)
end = timer()
etime = end - start
etime_h = etime//3600
etime_min = (etime % 3600)//60
etime_s = (etime % 3600) % 60
print(f"Total run time: {etime_h:.0f}h {etime_min:.0f}\' {etime_s:.0f}\"")