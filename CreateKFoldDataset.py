import json
import math
import numpy as np
import os
import random
from argparse import ArgumentParser


root_path = './damd_multiwoz/data'
dataset_path = os.path.join(root_path,'multi-woz-processed/data_for_damd.json')

gen_data_for_damd_reward = 'data_for_damd_reward_{}.json'
gen_reward_list_path = 'rewardListFile_{}_{}.json'

val_list_path=os.path.join(root_path,'multi-woz/valListFile.json')
test_list_path=os.path.join(root_path,'multi-woz/testListFile.json')

train_dataset_path = os.path.join(root_path,'multi-woz-processed',gen_data_for_damd_reward)
reward_list_path = os.path.join(root_path,'multi-woz-processed',gen_reward_list_path)


def get_list(file_path):
    fns = []
    with open(file_path,'r') as f:
        for line in f:
            fns.append(line.replace('\n','').replace('.json','').lower())
    return fns

def _get_reward_fns(safe_fns, NUM_PRED_SAMPLES_PER_JOB):
    while True:
        batch_fns = []
        random.shuffle(safe_fns)
        for fn in safe_fns:
            batch_fns.append(fn)
            if len(batch_fns)>=NUM_PRED_SAMPLES_PER_JOB:
                yield batch_fns
                batch_fns = []
                
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", dest="seed",
                        default=111,
                        type=int,
                        help="seed")
    parser.add_argument("-K", "--folds",
                        dest="K", default=10,
                        type=int,
                        help="Number of folds")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    NUM_JOBS = args.K
    
    
    valset_fns=get_list(val_list_path)
    testset_fns=get_list(test_list_path)
    
    print('Loading from:',dataset_path)
    all_dataset=json.loads(open(dataset_path,'r').read())
    
    safe_fns = all_dataset.keys()
    safe_fns = [k for k in safe_fns if k not in testset_fns]
    safe_fns = [k for k in safe_fns if k not in valset_fns]
    
    print('safe_fns:',len(safe_fns))
    
    TOTAL_NUM_SAMPLES = len(safe_fns)
    TRAIN_REWARD_SPLIT =  1 - 1/NUM_JOBS
    NUM_PRED_SAMPLES_PER_JOB = math.ceil(TOTAL_NUM_SAMPLES*(1-TRAIN_REWARD_SPLIT))
    NUM_TRAIN_SAMPLES_PER_JOB = math.ceil(TOTAL_NUM_SAMPLES*TRAIN_REWARD_SPLIT)
    
    print('TRAIN_REWARD_SPLIT;', TRAIN_REWARD_SPLIT)
    
    reward_tain_dataset_path = train_dataset_path.format(NUM_JOBS)
    # Removing original val-set
    frational_train_plus_test_dataset = {k:v for k,v in all_dataset.items() if k not in valset_fns}
    
    print('Total # of dialogues per fold:',len(frational_train_plus_test_dataset))
    print('K:',NUM_JOBS)
    print('# of train dialogues per fold:',NUM_TRAIN_SAMPLES_PER_JOB)
    print('# of val dialogues per fold:',NUM_PRED_SAMPLES_PER_JOB)
    print('reward_tain_dataset_path;',reward_tain_dataset_path)
    print('------------------------------------------------------------')
    
    with open(reward_tain_dataset_path,'w') as f:
        json.dump(frational_train_plus_test_dataset,f)
    
    reward_fns_itr = _get_reward_fns(safe_fns, NUM_PRED_SAMPLES_PER_JOB)
    for i in range(NUM_JOBS):
        reward_fns = next(reward_fns_itr)
        rewardListPath = reward_list_path.format(NUM_JOBS, i)
        print('fold-{} valset path:{}'.format(i,rewardListPath))
        with open(rewardListPath,'w') as f:
            for fn in reward_fns:
                f.write(fn.upper()+'.json\n')
