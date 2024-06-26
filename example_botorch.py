import matplotlib.pyplot as plt
import numpy as np
from hpob_handler import HPOBHandler
from methods.botorch import GaussianProcess
from methods.random_search import RandomSearch

valid_acquisitions = ["UCB", "EI", "PI"] #, "PM", "qEI"
seeds = ["test0", "test1", "test2", "test3", "test4"]
acc_list = []
n_trials = 50

hpob_hdlr = HPOBHandler(root_dir="hpob-data/", mode="v3-test")
search_space_id =  hpob_hdlr.get_search_spaces()[0]
dataset_id = hpob_hdlr.get_datasets(search_space_id)[0]

for acq_name in valid_acquisitions:
    acc_per_method = []
    for seed in seeds:
        print("Using ", acq_name, " as acquisition function...")

        #define the HPO method
        method = GaussianProcess(acq_name=acq_name)

        #evaluate the HPO method
        acc = hpob_hdlr.evaluate(method, search_space_id = search_space_id, 
                                                dataset_id = dataset_id,
                                                seed = seed,
                                                n_trials = n_trials )

        acc_per_method.append(acc)

    plt.plot(np.array(acc_per_method).mean(axis=0))
plt.legend(valid_acquisitions)
plt.xlabel('Trials', fontdict={'fontsize':12})
plt.ylabel('Accuracy', fontdict={'fontsize':12})
plt.title('GP',fontdict={'fontsize':17})
plt.savefig('plots/GP_acc.png')
plt.show()
