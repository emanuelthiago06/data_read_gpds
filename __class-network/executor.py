from class_based_network import Rede

def runner():
    rede = Rede()
    rede.set_val_path = ("caminho da validação")
    rede.set_train_path = ("caminho da treino")
    rede.set_test_path = ("caminho do teste")
    rede.set_executions_per_trials(number=3)
    rede.set_max_trials(number=3)
    rede.set_epochs(number = 40)
    rede.set_epochs_search(number = 4)
    rede.run_model_n_times(number = 2,save_results=True, file_name="test_with_2_loop.txt")
    rede.print_parameters()

if __name__ == "__main__":
    runner()