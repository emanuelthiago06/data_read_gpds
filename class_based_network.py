from django import db
from libimports import *

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
        self.__max_trials = 10
        self.__executions_per_trial = 10
    def __load_csv(self,path):
        try:
            db = pd.read_csv(path)
            return db
        except:
            raise SystemError(f"Erro na hora de carregar os arquivos, cheque se o caminho está correto ou se os arquivos tem a extensão csv: {path} ")
    def change_files(self, bolean: bool = True) -> None:
        """Não use esse método a não ser que você tenha lido a documentação e saiba oque ele faz"""
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

    def __split_val_test_train(self) -> None:
        self.__x_train, self.__y_train, self.__x_test, self.__y_test, self.__x_val, self.__y_val = split_pipeline(self.__clean_df_all, split_size = 0.15, categories_qtd = 2)
        self.__y_train = self.__y_train[:,0]
        self.__y_val = self.__y_val[:,0]
        self.__y_test = self.__y_test[:,0]

    def __model_execution(self) -> None:
        build_model(keras_tuner.HyperParameters(),self.__x_train.shape[1])
        tuner = keras_tuner.BayesianOptimization(
            hypermodel=build_model,
            objective="val_accuracy",
            max_trials=self.__max_trials,
            executions_per_trial=self.__executions_per_trial,
            overwrite=True,
            directory="my_dir",
            project_name="helloworld",
            )
        tuner.search(self.__x_train, self.__y_train, epochs=10, validation_data=(self.__x_val, self.__y_val))
        models = tuner.get_best_models(num_models=2)
        best_model = models[0]
        best_model.build(input_shape=(self.__x_train.shape[1],))
        best_model.summary()
        print("passou")
        tuner.results_summary(num_trials = 10)

    def set_drop(self,drop_list: list = [""]) -> None:
        """ Define as colunas que irão ser dropadas para fazer o dataframe útil """
        self.__drop_names = drop_list
    
    def set_val_path(self, path : str) -> None:
        """ Define o caminho do dataset da validação """
        self.__val_path = path

    def set_train_path(self, path : str) -> None:
        """ Define o caminho do dataset do treino """
        self.__train_path = path

    def set_test_path(self, path : str) -> None:
        """ Define o caminho do dataset do teste """
        self.__test_path = path

    def set_max_trials(self, number : int) -> None:
        """ Define o valor do max_trials do keras tuner que define a quantidade de vezes
        que o keras tenta achar os melhores valores """
        if not isinstance(number,int):
            raise SystemError("max-trials deve ser um numero inteiro")
        self.__max_trials = number

    def set_executions_per_trials(self, number : int) -> None:
        """ Define o valor do executions_per_trial do keras tuner, que define a quantidade de vezes
        que o keras vai rodar dentro de uma tentativa para achar os melhores parâmetros """
        if not isinstance(number,int):
            raise SystemError("executions-per-trial deve ser um numero inteiro")
        self.__executions_per_trial = number

    def set_class_elements(self, elements_list : list) -> None:
        """ Define os elementos dentro da sua classe é usado para filtrar, é necessário mandar
        uma lista que contém os elementos da sua classe ex: [1,0]"""
        self.__class_n = elements_list

    def run_model(self) -> None:
         self.__create_clean_df()
         self.__split_val_test_train()
         self.__model_execution()

    def show_df(self) -> None:
        self.__create_clean_df()
        self.__clean_df_all[1].head(10)

        
if __name__ == "__main__":
    rede = Rede()
    rede.change_files()
    rede.show_df()
    rede.run_model()

