from django import db
from libimports import *
from func import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SPLITS = ["train,val,test"]

class Rede:
    def __init__(self, **kwargs):
        self.__base_path = "/home/gpds/Documentos/ptb_vcg/dissertacao-main/resultados_csv" if "base_path" not in kwargs else kwargs["base_path"]
        self.__method = "Results_ST" if "method" not in kwargs else kwargs["method"]
        self.__val_name_age_sex = "dados_val_ptb_with_age_sex.csv" if "val_name_age_sex" not in kwargs else kwargs["val_name_age_sex"]
        self.__val_name = "dados_val_ptb.csv" if "val_name" not in kwargs else kwargs["val_name"]
        self.__train_name_age_sex = "dados_training_ptb_with_age_sex.csv" if "train_name_age_sex" not in kwargs else kwargs["train_name_age_sex"]
        self.__train_name = "dados_training_ptb.csv" if "train_name" not in kwargs else kwargs["train_name"]
        self.__test_name_age_sex = "dados_test_ptb_with_age_sex.csv" if "test_name_age_sex" not in kwargs else kwargs["test_name_age_sex"]
        self.__test_name = "dados_test_ptb.csv" if "test_name" not in kwargs else kwargs["test_name"]
        self.__drop_names = ['signalName', 'Unnamed: 0','split']
        self.__drop_names_age_sex = ['signalName', 'Unnamed: 0','split','Unnamed: 0.1','age']
        self.__class_name = "classe"
        self.__val_path = os.path.join(self.__base_path,self.__method, self.__val_name)
        self.__train_path = os.path.join(self.__base_path,self.__method, self.__train_name)
        self.__test_path = os.path.join(self.__base_path,self.__method, self.__test_name)
        self.__val_age_sex_path = os.path.join(self.__base_path,self.__method, self.__val_name_age_sex)
        self.__train_age_sex_path = os.path.join(self.__base_path,self.__method, self.__train_name_age_sex)
        self.__test_age_sex_path = os.path.join(self.__base_path,self.__method, self.__test_name_age_sex)
        self.__files_status = False
        self.__global_df = []
        self.__class_n = [1,0]
        self.__clean_df_all = []
    def __load_csv(self,path):
        try:
            db = pd.read_csv(path)
            return db
        except:
            raise SystemError(f"Erro na hora de carregar os arquivos, cheque se o caminho está correto ou se os arquivos tem a extensão csv: {path} ")
    def change_files(self, bolean: bool = True) -> None:
        self.__files_status = bolean

    def __define_global_df_and_drop(self) -> None:
        if self.__files_status:
            global_df = [self.__load_csv(self.__val_age_sex_path),self.__load_csv(self.__test_age_sex_path),self.__load_csv(self.__train_age_sex_path)]
            self.__drop_names = self.__drop_names_age_sex
        else:
            global_df = [self.__load_csv(self.__val_path),self.__load_csv(self.__test_path),self.__load_csv(self.__train_path)]
        self.__global_df = global_df

    def __split_by_class(self,df,target):
        new_df = df.loc[df[self.__class_name] == target].copy()
        return new_df

    def __create_clean_df(self) -> None:
        self.__define_global_df_and_drop()
        for df in self.__global_df:
            clean_df = df
            clean_df = clean_df.drop(columns= self.__drop_names) 
            clean_df = clean_df.loc[clean_df['classe'].isin(self.__class_n)]
            norm_df = self.__split_by_class(clean_df,target = 0)
            susp_df = self.__split_by_class(clean_df,target = 1)
            print("NORM --------- : ", norm_df.shape)
            print("SUSP ---------- : ", susp_df.shape)
            neg, pos = np.bincount(clean_df['classe'])
            total = neg + pos
            print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
                total, pos, 100 * pos / total))
            weight_for_0 = (1 / neg) * (total / 2.0)
            weight_for_1 = (1 / pos) * (total / 2.0)
            class_weight = {0: weight_for_0, 1: weight_for_1}

            print('Weight for class 0: {:.2f}'.format(weight_for_0))
            print('Weight for class 1: {:.2f}'.format(weight_for_1))

            self.__clean_df_all.append(clean_df)

    def set_drop(self,drop_list: list = [""]) -> None:
        self.__drop_names = drop_list
    
    def show_df(self):
        self.__create_clean_df()
        self.__clean_df_all[1].head(10)

        


