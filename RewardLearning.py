from gensim.models.keyedvectors import KeyedVectors
import json
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
import os
from random import shuffle
import re
import time
from tqdm import tqdm
import traceback

import numpy as np
import pandas as pd
from argparse import ArgumentParser
import random


class RewardLearning():
    
    def __init__(self, fold, seed, action_space, metric):
        self.reward_report_template = 'reward_report_{}_{}_.*.csv'
        word_embed_file_path='./damd_multiwoz/data/embeddings/glove.6B.100d.w2v.txt'
        
        self.train_val_fraction=0.8
        self.EMBED_DIM=100
        self.HIDDEN_DIM=100
        self.MAX_POP=10
        self.MAX_TIME_STEP=30
        self.MAX_GOAL_LEN=50
        self.MAX_STATE_LEN=50
        self.MAX_ACT_LEN=50
        self.reduce_lr_patience = 10
        self.es_patience = 25
        self.train_reward_split=[0.8,0.9][1]
        
        self.batch_size = 50
        self.num_epoch = 100
        
        self.fold = fold
        self.metric = metric
        self.TRAIN_ON=action_space
        
        self.root_path = './damd_multiwoz'
        self.dataset=json.loads(open(os.path.join(self.root_path,'data/multi-woz-processed/data_for_damd_reward_{}.json'.format(self.fold)),'r').read())
        self.glove_kv = KeyedVectors.load_word2vec_format(word_embed_file_path, binary=False, unicode_errors='ignore')

        self.reward_folder_path= os.path.join(self.root_path,'data/multi-woz-oppe/reward')
        self.data_for_damd = json.loads(open(os.path.join(self.root_path,'data/multi-woz-processed/data_for_damd.json'), 'r').read())
        
        self.processed_reward_rollouts = None
        self.embed_cache = {}



    def metric_score(self, sucess,match,bleu):
        return sucess+match+2*bleu/100
        
    def load_reward_rollouts(self):
        reward_record_file_prefix = self.reward_report_template.format(self.fold, self.metric)
        print('reward_record_file_prefix:',reward_record_file_prefix)
        rollouts_processed = {}
        for file in os.listdir(self.reward_folder_path):
            if re.search(reward_record_file_prefix,file):
                print('file:',file)
                reward_record_path = os.path.join(self.reward_folder_path,file)
                df = pd.read_csv(reward_record_path)
                for _,row in df.iterrows():
                    dial_id = row['dial_id']
                    rollout = json.loads(row['rollout'])
                    turn_nums = [int(z) for z in rollout.keys()]
                    turn_nums = sorted(turn_nums)
        
                    if dial_id not in rollouts_processed:
                        rollouts_processed[dial_id]={}
                        rollouts_processed[dial_id]['gen']=[]
                    
                    dia_rollout={}
                    rollouts_processed[dial_id]['gen'].append(dia_rollout)
                    dia_rollout['score'] = self.metric_score(row['success'],row['match'],row['bleu'])
                    
                    dia_rollout['rollout']=[]
                    for turn_num in turn_nums:
                        true_act_prob = [1.]
                        if 'aspn_prob' in rollout[str(turn_num)]:
                            true_act_prob = np.exp(rollout[str(turn_num)]['aspn_prob']).tolist()
                        dia_rollout['rollout'].append({
                            'tn':turn_num,
                            'act':rollout[str(turn_num)]['aspn_gen'],
                            'true_act':rollout[str(turn_num)]['aspn'],
                            'resp':rollout[str(turn_num)]['resp_gen'],
                            'true_act_prob':true_act_prob
                        })
                    
                        
                    if 'gt' not in rollouts_processed[dial_id]:
                        rollouts_processed[dial_id]['gt']={}
                        rollouts_processed[dial_id]['gt']['score']=4
                        rollouts_processed[dial_id]['gt']['rollout']=[]
                        for turn_num in turn_nums:
                            rollouts_processed[dial_id]['gt']['rollout'].append({
                                'tn':turn_num,
                                'act':rollout[str(turn_num)]['aspn'],
                                'resp':rollout[str(turn_num)]['resp'],
                                'true_act':rollout[str(turn_num)]['aspn'],
                                'true_act_prob':[1]
                            })
                            
        self.processed_reward_rollouts = rollouts_processed
        self.dial_ids = list(self.processed_reward_rollouts.keys())
        self.load_gt_dia_logs(self.dial_ids)
        return rollouts_processed

    def load_gt_dia_logs(self, dial_ids):
        gt_dia_logs={}
        for dial_id in dial_ids:
            goal = self.goal_as_st(self.dataset[dial_id]['goal'])
            gt_dia_log={
                'goal':goal
            }
            gt_dia_logs[dial_id]=gt_dia_log
            for turn in self.dataset[dial_id]['log']:
                gt_dia_log[turn['turn_num']]={}
                gt_dia_log[turn['turn_num']]['state']='begin '+turn['cons_delex']+' end'
                
        self.gt_dia_logs = gt_dia_logs       
                    
    def pad_sentence(self, token_embeds,max_seq_len):
        token_embeds = token_embeds.copy()
        token_embeds = token_embeds[:max_seq_len].tolist()
        
        for i in range(max_seq_len-len(token_embeds)):
            token_embeds.append(np.zeros(self.EMBED_DIM))
        
        token_embeds = np.array(token_embeds)
        return token_embeds
    
    def pad_time_step(self, sentence_embeds,max_seq_len):
        sentence_embeds = sentence_embeds[:self.MAX_TIME_STEP]
        time_padded_sentences = np.array(sentence_embeds)
        if self.MAX_TIME_STEP>len(sentence_embeds):
            pad = np.zeros((self.MAX_TIME_STEP-len(sentence_embeds),max_seq_len,self.EMBED_DIM))
            time_padded_sentences = np.concatenate([sentence_embeds,pad])
        return time_padded_sentences
        
    def get_embedding(self, token):
        token = token.lower()
        token = token.replace('reqt','request')\
                .replace('arriveby','arrive_by')\
                .replace('towninfo','town_info')\
                .replace('pricerange','price_range')\
                .replace('leaveat','leave_at')\
                .replace('mutliple','multiple')\
                .replace('dontcare','dont_care')\
                .replace('-','')\
                .replace('addres','address')\
                .replace('addressss','address')\
                .replace('addresss','address')
        token = token.strip()
        if token in self.embed_cache:
            return self.embed_cache[token]
        if token in self.glove_kv:
            embedding = self.glove_kv[token]
        else:
            if '_' in token:
                embeds = []
                for sub_token in token.split('_'):
                    embeds.append(self.get_embedding(sub_token))
                embedding = np.mean(embeds,axis=0)
            else:
                #print('token not in embed:',token)
                embedding = self.glove_kv['unk']
        self.embed_cache[token]=embedding
        return embedding
    
    def tokens_to_embeddings(self, tokens):
        embeddings = []
        for token in tokens:
            embeddings.append(self.get_embedding(token))
        return np.array(embeddings)
    
    def tokenize(self, sentence):
        sentence=sentence.lower()
        sentence = sentence.replace('[',' ').replace(']',' ').replace(':','').replace('  ',' ')
        
        return sentence.split()
    
    def goal_as_st(self, goal):
        return str(goal).replace("'",' ')\
                        .replace(',',' , ').replace('{',' ')\
                        .replace('}',' ').replace('  ',' ')
    
    def sample_roll_out(self, dial_id):
        start = time.time()
        gen_rollouts_info = self.processed_reward_rollouts[dial_id]['gen']
        gt_rollout_info = self.processed_reward_rollouts[dial_id]['gt']
        rollout_infos = np.random.choice(gen_rollouts_info+[gt_rollout_info], size=2, replace=False)
        #print(rollout_infos)
        
        dia_log= self.gt_dia_logs[dial_id]
        goal = dia_log['goal']
        
        goal = self.tokenize(goal)
        goal = self.tokens_to_embeddings(goal)
        goal = self.pad_sentence(goal, self.MAX_GOAL_LEN)
        
        rollout_pairs = []
        for rollout_info in rollout_infos:
            acts = []
            states = []
            for turn in rollout_info['rollout']:
                tn = turn['tn']
                act = turn[self.TRAIN_ON]#turn['act']
                
                if tn not in self.gt_dia_logs[dial_id]:
                    break
                
                state = self.gt_dia_logs[dial_id][tn]['state']
                
#                 if random.uniform(0,1)>0.95:
#                     print('act:',act)
#                print('state:',state)
                act = self.tokenize(act)
                state = self.tokenize(state)
                
                act = self.tokens_to_embeddings(act)
                state = self.tokens_to_embeddings(state)
                
                act = self.pad_sentence(act,self.MAX_ACT_LEN)
                state = self.pad_sentence(state,self.MAX_STATE_LEN)
        
                acts.append(act)
                states.append(state)
            
            acts=self.pad_time_step(acts,self.MAX_ACT_LEN)
            states=self.pad_time_step(states,self.MAX_STATE_LEN)
        
            score=rollout_info['score']
            rollout_pairs.append([goal,states,acts,score])
        prob = rollout_pairs[0][-1]/(rollout_pairs[0][-1]+rollout_pairs[1][-1]+1e-20)
        rollout_pairs[0][-1]=prob
        rollout_pairs[1][-1]=1-prob
        
        return rollout_pairs
    
    def get_data_gen(self, sample_roll_out):
        def data_gen(dial_ids,batch_size):
            try:
                s1s = []
                a1s = []
                g1s = []
    
                s2s = []
                a2s = []
                g2s = []
    
                probs = []
                while True:
                    shuffle(dial_ids)
                    for dial_id in dial_ids:
                        rollout_pair = sample_roll_out(dial_id)
                        g1,s1,a1,p1=rollout_pair[0]
                        g2,s2,a2,p2=rollout_pair[1]
    
                        s1s.append(s1)
                        a1s.append(a1)
                        g1s.append(g1)
                        s2s.append(s2)
                        a2s.append(a2)
                        g2s.append(g2)
    
                        probs.append([p1,p2])
    
                        if len(s1s)>=batch_size:
                            s1s = np.array(s1s)
                            a1s = np.array(a1s)
                            g1s = np.array(g1s)
    
                            s2s = np.array(s2s)
                            a2s = np.array(a2s)
                            g2s = np.array(g2s)
    
                            #print('as:',np.sum(a1s-a2s))
    
                            probs = np.array(probs)
                            yield [s1s,a1s,g1s,s2s,a2s,g2s],probs
                            s1s = []
                            a1s = []
                            g1s = []
    
                            s2s = []
                            a2s = []
                            g2s = []
    
                            probs = []
                            
            except Exception as e:
                print(traceback.format_exc())
                raise e
    
        return data_gen
    
    
    def build_reward_model(self):
        s_bilstm = Bidirectional(LSTM(self.HIDDEN_DIM)) 
        a_bilstms = [Conv1D(self.HIDDEN_DIM,1,activation='tanh'),
                     Conv1D(self.HIDDEN_DIM,1,activation='tanh'),
                     Lambda(lambda z:K.mean(z,axis=-2))]
        a_bilstms=[Bidirectional(LSTM(self.HIDDEN_DIM))] 
        g_bilstm = Bidirectional(LSTM(self.HIDDEN_DIM)) 
        
        
        reward_convs=[]
        reward_convs.append(Dense(self.HIDDEN_DIM,activation='tanh'))
        reward_convs.append(Dense(self.HIDDEN_DIM,activation='tanh'))
        reward_convs.append(Dense(self.HIDDEN_DIM,activation='tanh'))
        reward_convs.append(Dense(1,activation='sigmoid'))
        
        s = Input(shape=(self.MAX_STATE_LEN, self.EMBED_DIM))
        a = Input(shape=(self.MAX_ACT_LEN, self.EMBED_DIM))
        g = Input(shape=(self.MAX_GOAL_LEN, self.EMBED_DIM))
        
        s_h = s_bilstm(s)
        a_h = a
        for layer in a_bilstms:
            a_h = layer(a_h)
        g_h = g_bilstm(g)
        
        #s_h = Lambda(lambda z:z*1e-20)(s_h)
        #g_h = Lambda(lambda z:z*1e-20)(g_h)
        
        reward = Concatenate(axis=-1)([s_h,a_h,g_h])
        for reward_conv in reward_convs:
            reward = reward_conv(reward)
        reward = Lambda(lambda z:K.squeeze(z,axis=-1))(reward)
        
        model_reward = Model(inputs=[s,a,g],outputs=reward)
        model_reward.summary()
        return model_reward
    
    def _build_reward_flatten_model(self):
        x = Input(shape=(self.MAX_STATE_LEN + self.MAX_ACT_LEN + self.MAX_GOAL_LEN, self.EMBED_DIM))
        s=Lambda(lambda z:z[:,:self.MAX_STATE_LEN])(x)
        a=Lambda(lambda z:z[:,self.MAX_STATE_LEN : self.MAX_STATE_LEN + self.MAX_ACT_LEN])(x)
        g=Lambda(lambda z:z[:,self.MAX_STATE_LEN + self.MAX_ACT_LEN:])(x)
        
        reward = self.model_reward([s,a,g])
        model_reward_flatten = Model(x,reward)
        model_reward_flatten.summary()
        return model_reward_flatten
    
    def _build_cummulative_reward_model(self):
        
        model_reward_flatten = self._build_reward_flatten_model()
        
        s = Input(shape=(self.MAX_TIME_STEP, self.MAX_STATE_LEN, self.EMBED_DIM))
        a = Input(shape=(self.MAX_TIME_STEP, self.MAX_ACT_LEN, self.EMBED_DIM))
        g = Input(shape=(self.MAX_GOAL_LEN, self.EMBED_DIM))
        
        g_padded = Lambda(lambda z:K.expand_dims(z,axis=1))(g)
        g_padded = Lambda(lambda z:K.repeat_elements(z, self.MAX_TIME_STEP,axis=1))(g_padded)
        
        comb_inp = Concatenate(axis=2)([s,a,g_padded])
        
        rewards = TimeDistributed(model_reward_flatten)(comb_inp)
        
        
        returns = Lambda(lambda z:K.sum(z,axis=1,keepdims=True))(rewards)
            
        model_cummulative_reward = Model([s,a,g],returns)
        model_cummulative_reward.summary()
        return model_cummulative_reward
    
    def _build_preferential_model(self):
        
        model_cummulative_reward = self._build_cummulative_reward_model()
        
        s_1 = Input(shape=(self.MAX_TIME_STEP, self.MAX_STATE_LEN, self.EMBED_DIM))
        a_1 = Input(shape=(self.MAX_TIME_STEP, self.MAX_ACT_LEN, self.EMBED_DIM))
        g_1 = Input(shape=(self.MAX_GOAL_LEN, self.EMBED_DIM))
        
        s_2 = Input(shape=(self.MAX_TIME_STEP, self.MAX_STATE_LEN, self.EMBED_DIM))
        a_2 = Input(shape=(self.MAX_TIME_STEP, self.MAX_ACT_LEN, self.EMBED_DIM))
        g_2 = Input(shape=(self.MAX_GOAL_LEN, self.EMBED_DIM))
        
        chi_1 = model_cummulative_reward([s_1,a_1,g_1])
        chi_2 = model_cummulative_reward([s_2,a_2,g_2])
        
        chi = Concatenate()([chi_1,chi_2])
        #Pref = Activation('softmax')(chi)
        Pref = Lambda(lambda z:z/K.sum(z,axis=-1,keepdims=True))(chi)
        
        model_preferential = Model([s_1,a_1,g_1,s_2,a_2,g_2],Pref)
        model_preferential.summary()
        return model_preferential
    
    
    
    def get_reward(self, input_seq):
        g = []
        s = []
        a = []
        for goal,state, aspn, resp in input_seq:
        
            state_tokens = self.tokenize(state)
            state_token_embeds = self.tokens_to_embeddings(state_tokens)
            state_token_embeds = self.pad_sentence(state_token_embeds, self.MAX_STATE_LEN)
            s.append(state_token_embeds)
    
            if self.TRAIN_ON=='act':
                action_tokens = self.tokenize(aspn)
            elif self.TRAIN_ON=='resp':
                action_tokens = self.tokenize(resp)
            else:
                raise Exception('Invalid TRAIN_ON selection')
            action_token_embeds = self.tokens_to_embeddings(action_tokens)
            action_token_embeds = self.pad_sentence(action_token_embeds, self.MAX_ACT_LEN)
            a.append(action_token_embeds)
    
            goal_tokens = self.tokenize(goal)
            goal_token_embeds = self.tokens_to_embeddings(goal_tokens)
            goal_token_embeds = self.pad_sentence(goal_token_embeds, self.MAX_GOAL_LEN)
            g.append(goal_token_embeds)
        
        rewards = self.model_reward.predict([np.array(s),np.array(a),np.array(g)])
        #print('aspn:',aspn,':',reward)
        
        return rewards
    
    def get_Gs(self,  gamma=0.9):
        fn_Gs = {}
        num_fns = len(self.data_for_damd.keys())
        for ex_num,fn in enumerate(tqdm(reversed(list(self.data_for_damd.keys())),total=num_fns)):
            #print('%:{0.2f}'.format(ex_num/num_fns),end='')
            next_state=None
            
            fn_Gs[fn] = {}
            goal = self.goal_as_st(self.data_for_damd[fn]['goal'])
            
            turn_num_inp_seq = {}
            for turn in self.data_for_damd[fn]['log']:
                turn_num = turn['turn_num']
                resp = turn['resp']
                state = 'begin '+turn['cons_delex']+' end'#turn['cons_delex']
                aspn = turn['sys_act']
                
                turn_num_inp_seq[turn_num]=[goal,state,aspn,resp]
                
            reverse_turn_nums = sorted(list(turn_num_inp_seq.keys()),reverse=True)
            inp_seq = []
            for turn_num in reverse_turn_nums:
                inp_seq.append(turn_num_inp_seq[turn_num])
                
            rewards = self.get_reward(inp_seq)
            G = 0
            for turn_num,reward in zip(reverse_turn_nums,rewards):
                G = reward + gamma*G
                fn_Gs[fn][turn_num] = {
                    'G':G,
                    'gamma':gamma
                }
        return fn_Gs

    def compile_models(self):
        self.model_reward = self.build_reward_model()
        self.model_preferential = self._build_preferential_model()
        self.model_preferential.compile(loss='categorical_crossentropy', optimizer='adam')

    def train_model(self):
        shuffle(self.dial_ids)
        train_dial_ids = self.dial_ids[:int(len(self.dial_ids) * self.train_val_fraction)]
        val_dial_ids = self.dial_ids[int(len(self.dial_ids) * self.train_val_fraction):]
        
        train_num_examples = len(train_dial_ids)
        valid_num_examples = len(val_dial_ids)

        print('train_num_examples:',train_num_examples)
        print('valid_num_examples:',valid_num_examples)
        
        train_num_examples_per_epoch = max(3,int((train_num_examples/self.batch_size)/10))
        
        train_data_gen = self.get_data_gen(self.sample_roll_out)(train_dial_ids, self.batch_size)
        val_data_gen = self.get_data_gen(self.sample_roll_out)(val_dial_ids, self.batch_size)
        
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=self.reduce_lr_patience, min_lr=0.000001,verbose=1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=self.es_patience, verbose=1, restore_best_weights=True)

        self.model_preferential.fit_generator(train_data_gen,
                               steps_per_epoch = train_num_examples_per_epoch, 
                               validation_data = val_data_gen,
                               validation_steps = max(1,int(valid_num_examples/(self.batch_size))),
                               callbacks = [reduce_lr,early_stopping],
                               epochs = self.num_epoch, 
                              )

    def save_returns(self, gamma=0.):
        num_fns = len(self.data_for_damd.keys())
        fn_Gs = self.get_Gs(gamma=gamma)
        fn_G_file_name = 'fn_Gs_{}_{}_{}_{}.json'.format(self.fold, gamma, self.TRAIN_ON, self.metric)
        
        print(fn_G_file_name)
        fn_Gs_file_path = os.path.join(self.root_path,'data','multi-woz-oppe',fn_G_file_name)
        print('fn_Gs_file_path:',fn_Gs_file_path)
        with open(fn_Gs_file_path,'w') as f:
            json.dump(fn_Gs,f)

    
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
    
    print('param:',args)       
    rewardLearning = RewardLearning(args.folds, args.seed, args.action_space, args.metric)
    rewardLearning.load_reward_rollouts()
    rewardLearning.compile_models()
    rewardLearning.train_model()
    rewardLearning.save_returns(args.gamma)






    







