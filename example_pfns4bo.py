import torch
import matplotlib.pyplot as plt
import numpy as np
from hpob_handler import HPOBHandler
# from methods.pfns.pfns4bo import pfns4bo
from methods.pfns.scripts.acquisition_functions import TransformerBOMethod

model_path = 'C:/Users/82109/dsl_lab/HPO-B_taejoo/methods/pfns/final_models/model_sampled_warp_simple_mlp_for_hpob_46.pt'
# pfn_bo = TransformerBOMethod(torch.load(model_path), device='cpu:0', acq_function='pi')

valid_acquisitions = ["ucb", "ei", "pi"] #"PM", "qEI", "hebo"
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
        method = TransformerBOMethod(torch.load(model_path), device='cpu:0', acq_function=acq_name)

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
plt.title('Pretrained PFNs4BO',fontdict={'fontsize':17})
plt.savefig('plots/pretrained_PFNs4B0_acc.png')
plt.show()
