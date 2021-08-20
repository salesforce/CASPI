from collections import defaultdict
import json
import os
import random

import numpy as np
from argparse import ArgumentParser


K=10
TRAIN_ON=['act','resp'][0]
GAMMA_GLOBAL = 0.0
USE_R_AS_G = True
METRIC=['soft', 'hard'][0]

def get_turn_state(full_state,turn_domain):
    all_domain = ['[police]', '[taxi]', '[restaurant]', '[attraction]', '[hotel]', '[hospital]', '[train]', '[general]']
    if turn_domain not in full_state:
        return turn_domain
    turn_domain_idx = full_state.index(turn_domain)
    end_idx = len(full_state)
    for domain in all_domain:
        if turn_domain!=domain:
            if domain in full_state:
                domain_idx = full_state.index(domain)
                if domain_idx>turn_domain_idx and domain_idx<end_idx:
                    end_idx=domain_idx
                    
    return full_state[turn_domain_idx:end_idx].strip()

def get_state_act(data_for_damd):
    fn_tn_state = defaultdict(dict)
    state_act = defaultdict(list)
    act_state = defaultdict(list)
    for fn,dia in data_for_damd.items():
        if fn not in test_fn:
            prev_act = None
            for turn_num,log in enumerate(dia['log']): 
                full_state=log['cons_delex']
                turn_domains = log['turn_domain']
                turn_domains = turn_domains.split(' ')
                turn_domain = turn_domains[-1]
                if turn_domain not in full_state:
                    turn_state = turn_domain
                else:
                    turn_state = get_turn_state(full_state,turn_domain)
                if prev_act is not None:
                    act_state[prev_act].append(turn_state)
                act = log['sys_act']
                state_act[turn_state].append(act)
                fn_tn_state[fn][turn_num] = turn_state
                prev_act = act
    return state_act,act_state,fn_tn_state

def get_state(fn, fn_tn_state, tn):
    turn_state = fn_tn_state[fn][tn]
    return turn_state

def get_act(turn):
    return turn['sys_act']

def get_gamma(gamma_local):
    if GAMMA_GLOBAL is not None:
        gamma = GAMMA_GLOBAL
    else:
        gamma = gamma_local
    return gamma


not_in_fn_Gs = set()
def get_reward_gamma(fn_Gs, fn, turn_num):
    turn_num = str(turn_num)
    if fn not in fn_Gs:
        not_in_fn_Gs.add(fn)
        return None
    if USE_R_AS_G == False:
        reward = fn_Gs[fn][turn_num]['R']
        gamma = fn_Gs[fn][turn_num]['gamma']
    elif USE_R_AS_G == True:
        reward = fn_Gs[fn][turn_num]
        gamma = 0 
    else:
        raise Exception('Invalid USE_R_AS_G selection')
    return reward, gamma


def get_value_function(data_for_damd, fn_tn_state, fn_Gs):
    V_info = {}
    for fn, dia in data_for_damd.items():
        if fn not in test_fn:
            log = dia['log']
            G_nxt = 0
            for turn in reversed(log):
                turn_num = turn['turn_num']
                state = get_state(fn, fn_tn_state, turn_num)
                R_gamma = get_reward_gamma(fn_Gs ,fn, turn_num)
                if R_gamma is None:
                    continue
                R,gamma = R_gamma[0]['G'], R_gamma[0]['gamma']
                if state not in V_info:
                    V_info[state] = {
                        'V':0,
                        '|S|':0
                    }
                    if USE_R_AS_G == False:
                        G = R + get_gamma(gamma) * G_nxt
                    elif USE_R_AS_G == True:
                        G = R
                    else:
                        raise Exception('Invalid USE_R_AS_G selection')
                    V_info[state]['V'] = (V_info[state]['V'] * V_info[state]['|S|'] + G)/(V_info[state]['|S|']+1)
                    V_info[state]['|S|']+=1
                    G_nxt = G
    return V_info                
                  
    
def get_Q_function(data_for_damd, V_info, fn_tn_state, fn_Gs):
    Q_info = {}
    for fn, dia in data_for_damd.items():
        if fn not in test_fn:
            log = dia['log']
            V_nxt = 0
            for turn in reversed(log):
                turn_num = turn['turn_num']
                state = get_state(fn, fn_tn_state, turn_num)
                act = get_act(turn)
                R_gamma = get_reward_gamma(fn_Gs, fn, turn_num)
                if R_gamma is None:
                    continue
                R,gamma = R_gamma[0]['G'], R_gamma[0]['gamma']
                if state not in Q_info:
                    Q_info[state] = {}
                if act not in Q_info[state]:
                    Q_info[state][act] = {
                        'Q':0,
                        '|S|':0
                    }
                    G = R + get_gamma(gamma) * V_nxt
                    Q_info[state][act]['Q'] = (Q_info[state][act]['Q'] * Q_info[state][act]['|S|'] + G)/(Q_info[state][act]['|S|']+1)
                    Q_info[state][act]['|S|']+=1
                    V_nxt = V_info[state]['V']
    return Q_info  
 



def estimate_bh_policy(state_act, state, act):
    Z = len(state_act[state])
    P_act = state_act[state].count(act)/Z
    return P_act
    
def persist_Q_function(data_for_damds, Q_infos, state_acts, fn_tn_states, path_to_persist):
    Q_fn = {}
    for data_for_damd, Q_info, state_act , fn_tn_state in zip(data_for_damds, Q_infos, state_acts, fn_tn_states):
        for fn, dia in data_for_damd.items():
            if fn not in test_fn:
                log = dia['log']
                Q_fn[fn] = {}
                for turn in log:
                    turn_num = turn['turn_num']
                    state = get_state(fn, fn_tn_state, turn_num)
                    act = get_act(turn)
                    if state not in Q_info or act not in Q_info[state]:
                        raise Exception('I Dont see a reason to be here!')
                        Q_fn[fn][turn_num] = {
                            'Q':0,
                            'prob':1
                        }
                    else:
                        act_len = max(1,len(act.split()))
                        bh_policy = estimate_bh_policy(state_act, state, act)
                        
                        Q_fn[fn][turn_num] = {
                            'Q':Q_info[state][act]['Q'],
                            'prob':bh_policy
                        }
    if path_to_persist is not None:
        print('path_to_persist:',path_to_persist)
        with open(path_to_persist, 'w') as f:
            json.dump(Q_fn,f,indent=2)
    return Q_fn

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", dest="seed",
                        default=11,
                        type=int,
                        help="seed")
    parser.add_argument("-K", "--folds",
                        dest="folds", default=10,
                        type=int,
                        help="Number of folds")
    parser.add_argument("-a", "--action_space",
                        dest="action_space",
                        choices={"act", "resp"},
                        default='act',
                        help="action space. can either be act or resp")
    parser.add_argument("-m", "--metric",
                        dest="metric",
                        choices={"hard", "soft"},
                        default='soft',
                        help="metric used for pairwise reward candidate generation")
    parser.add_argument("-g", "--gamma",
                        dest="gamma",
                        default=0.0,
                        type=float,
                        help="The discount factor used in reward learning")
    args = parser.parse_args()
    
    
    
    args = parser.parse_args()
    
    K=args.folds
    TRAIN_ON=args.action_space
    GAMMA_GLOBAL = args.gamma
    METRIC = args.metric
    fn_G_file_name = 'fn_Gs_{}_{}_{}_{}.json'.format(K, GAMMA_GLOBAL, TRAIN_ON, METRIC)
    
    set_seed(args.seed)
    
    root_path = './damd_multiwoz/data'

    test_fn_json_path = os.path.join(root_path,'multi-woz/testListFile.json')
    valid_fn_json_path = os.path.join(root_path,'multi-woz/valListFile.json')

    test_fn = set(open(test_fn_json_path,'r').read().lower().replace('.json','').split())
    valid_fn = set(open(valid_fn_json_path,'r').read().lower().replace('.json','').split())

    data_for_damd = json.loads(open(os.path.join(root_path,'multi-woz-processed/data_for_damd.json'),'r').read())
    
    print(fn_G_file_name)
    fn_Gs_file_path = os.path.join(root_path,'multi-woz-oppe',fn_G_file_name)
    fn_Gs = json.loads(open(fn_Gs_file_path,'r').read())
    
    data_for_damd_only_train = {
        fn:v for fn,v in data_for_damd.items() if fn not in test_fn and fn not in valid_fn
    }
    print('Train filtered/unfiltered={}/{}'.format(len(data_for_damd_only_train),len(data_for_damd)))
    
    data_for_damd_only_val = {
        fn:v for fn,v in data_for_damd.items() if fn in valid_fn
    }
    print('Val filtered/unfiltered={}/{}'.format(len(data_for_damd_only_val),len(data_for_damd)))
    
    state_act_train,_,fn_tn_state_train = get_state_act(data_for_damd_only_train)
    state_act_val,_,fn_tn_state_val = get_state_act(data_for_damd_only_val)
    
    V_info_train = get_value_function(data_for_damd_only_train, fn_tn_state_train, fn_Gs)
    V_info_val = get_value_function(data_for_damd_only_val, fn_tn_state_val, fn_Gs)
    
    Q_info_train = get_Q_function(data_for_damd_only_train, V_info_train, fn_tn_state_train, fn_Gs)
    Q_info_val = get_Q_function(data_for_damd_only_val, V_info_val, fn_tn_state_val, fn_Gs)
    
    
    Q_fn_path_to_persist = os.path.join(root_path,'multi-woz-oppe',fn_G_file_name.replace('fn_Gs_','fn_Qs_'))
    
    
    Q_fn = persist_Q_function([data_for_damd_only_train, data_for_damd_only_val],
                          [Q_info_train, Q_info_val],
                          [state_act_train, state_act_val],
                          [fn_tn_state_train, fn_tn_state_val],
                          Q_fn_path_to_persist)
    

