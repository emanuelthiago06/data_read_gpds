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
        self.__train_name_age_sex = "dados_train_ptb_with_age_sex.csv" if "train_name_age_sex" not in kwargs else kwargs["train_name_age_sex"]
        self.__train_name = "dados_train_ptb.csv" if "train_name" not in kwargs else kwargs["train_name"]
        self.__test_name_age_sex = "dados_test_ptb_with_age_sex.csv" if "test_name_age_sex" not in kwargs else kwargs["test_name_age_sex"]
        self.__test_name = "dados_test_ptb.csv" if "test_name" not in kwargs else kwargs["test_name"]
        self.__drop_names = ['signalName', 'Unnamed: 0','split']
        self.__drop_names_age_sex = ['signalName', 'Unnamed: 0','split','Unnamed: 0.1','age']
        self.__class_name = "classe"
        self.__val_path = os.path.join(self.__base_path,self.__method, self.__val_name)
        self.__train_path = os.path.join(self.__base_path,self.__method, self.__train_name)
        self.__test_path = os.path.join(self.__base_path,self.__method, self.__test_name)
        self.__val_age_sex_path = os.path.join(self.__base_path,self.__method, self.__val_name)
        self.__train_age_sex_path = os.path.join(self.__base_path,self.__method, self.__train_name)
        self.__test_age_sex_path = os.path.join(self.__base_path,self.__method, self.__test_name)
        self.__files_status = False
        self.__global_df = []
    def __load_csv(self,path):
        try:
            db = pd.read_csv(path)
            return db
        except:
            raise SystemError("Erro na hora de carregar os arquivos, cheque se o caminho está correto ou se os arquivos tem a extensão csv")
    def change_files(self, bolean: bool = True) -> None:
        self.__files_status = bolean

    def __define_global_df_and_drop(self) -> None:
        if self.__files_status:
            global_df = [self.__load_csv(self.__val_age_sex_path),self.__load_csv(self.__test_age_sex_path),self.__load_csv(self.__train_age_sex_path)]
            self.__drop_names = self.__drop_names_age_sex
        else:
            global_df = [self.__load_csv(self.__val_path),self.__load_csv(self.__test_path),self.__load_csv(self.__train_path)]
        self.__global_df = global_df

    def __create_clean_df(self):
        for df in self.__global_df:
            pass

    def set_drop(self,drop_list: list = [""]) -> None:
        self.__drop_names = drop_list

        


