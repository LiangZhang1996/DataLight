from .generator import Generator
from . import model_test
import json
import shutil
import os
import time
import pickle
import numpy as np


def path_check(dic_path):
    if os.path.exists(dic_path["PATH_TO_WORK_DIRECTORY"]):
        if dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default":
            raise FileExistsError
        else:
            pass
    else:
        os.makedirs(dic_path["PATH_TO_WORK_DIRECTORY"])
    if os.path.exists(dic_path["PATH_TO_MODEL"]):
        if dic_path["PATH_TO_MODEL"] != "model/default":
            raise FileExistsError
        else:
            pass
    else:
        os.makedirs(dic_path["PATH_TO_MODEL"])


def copy_conf_file(dic_path, dic_agent_conf, dic_traffic_env_conf, path=None):
    if path is None:
        path = dic_path["PATH_TO_WORK_DIRECTORY"]
    json.dump(dic_agent_conf, open(os.path.join(path, "agent.conf"), "w"), indent=4)
    json.dump(dic_traffic_env_conf, open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)


def copy_cityflow_file(dic_path, dic_traffic_env_conf, path=None):
    if path is None:
        path = dic_path["PATH_TO_WORK_DIRECTORY"]
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf["TRAFFIC_FILE"]),
                os.path.join(path, dic_traffic_env_conf["TRAFFIC_FILE"]))
    shutil.copy(os.path.join(dic_path["PATH_TO_DATA"], dic_traffic_env_conf["ROADNET_FILE"]),
                os.path.join(path, dic_traffic_env_conf["ROADNET_FILE"]))


def generator_wrapper(cnt_round, cnt_gen, dic_path, dic_agent_conf, dic_traffic_env_conf, memory):
    generator = Generator(cnt_round=cnt_round,
                          cnt_gen=cnt_gen,
                          dic_path=dic_path,
                          dic_agent_conf=dic_agent_conf,
                          dic_traffic_env_conf=dic_traffic_env_conf,
                          )
    print("make generator")
    generator.train_model(memory, cnt_round)
    print("generator_wrapper end")
    return


def create_mask(seq_length, a):
    total_length = 3 * seq_length
    mask = np.ones((total_length, total_length), dtype=np.float32)
    for i in range(seq_length):
        start_idx = 3 * i
        end_idx = start_idx + 3
        mask[start_idx:end_idx, :end_idx] = 0
    mask[:, :3*a] = 1
    return mask


class Pipeline:

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path
        self.k_len = dic_traffic_env_conf["K_LEN"]
        self.t_num = dic_traffic_env_conf["T_NUM"]
        self.rtg = dic_traffic_env_conf["RTG"]
        self.initialize()
        
    def load_data(self):
        path1 = self.dic_path["PATH_TO_MEMORY"]
        print(path1)

        with open(path1, "rb") as f:
            memory = pickle.load(f)

        return memory

    def prepare_samples_DT(self, memory, num):
        """
        num: number of transitons per episode
        """
        state, action, next_state, p_reward, ql_reward, _, _, _, _ = memory
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"]
        memory_size = len(action)
        _state = [[], None]
        _next_state = [[], None]
        for feat_name in used_feature:
            if feat_name == "phase12":
                _state[1] = np.array(state[feat_name])
                _next_state[1] = np.array(next_state[feat_name])
            else:
                _state[0].append(np.array(state[feat_name]).reshape(memory_size, 12, -1))
                _next_state[0].append(np.array(next_state[feat_name]).reshape(memory_size, 12, -1))
          
        _state2, _next_state2 = [np.concatenate(_state[0], axis=-1), _state[1]], [np.concatenate(_next_state[0], axis=-1), _next_state[1]]
        _, _, b = _state2[0].shape
        N = int(memory_size/num)
        _state3, _next_state3 = [[] for _ in range(5)], [[] for _ in range(5)]
        mask = []
        tt1, tt2 = [], []
        R1, R2 = [], []
        for i in range(N):
            for j in range(num):
                idx = i*num + j
                if j+1 < self.k_len:
                    pad1, pad2 = np.zeros((self.k_len-j-1,12, b)), np.zeros((self.k_len-j-1,12))
                    tmp1, tmp2 = _state2[0][i*num:idx+1, :, :], _state2[1][i*num:idx+1, :]
                    tmp3, tmp4 = _next_state2[0][i*num:idx+1, :, :], _next_state2[1][i*num:idx+1, :]
                    tmp1, tmp2 = np.concatenate([pad1, tmp1], axis=0), np.concatenate([pad2, tmp2], axis=0)
                    tmp3, tmp4 = np.concatenate([pad1, tmp3], axis=0), np.concatenate([pad2, tmp4], axis=0)
                    tmp_mask = create_mask(self.k_len, self.k_len-j-1)
                else:
                    tmp1, tmp2 = _state2[0][idx-self.k_len+1:idx+1, :, :], _state2[1][idx-self.k_len+1:idx+1, :]
                    tmp3, tmp4 = _state2[0][idx-self.k_len+1:idx+1, :, :], _state2[1][idx-self.k_len+1:idx+1, :]
                    tmp_mask = create_mask(self.k_len, 0)
                if j==0:
                    rtg = self.rtg
                else:
                    rtg = rtg - p_reward[idx-1]   
                mask.append(tmp_mask)
                _state3[0].append(tmp1)
                _state3[1].append(tmp2)
                _next_state3[0].append(tmp3)
                _next_state3[1].append(tmp4) 
                tt1.append(j)
                tt2.append(j+1)
                R1.append(rtg)
                R2.append(rtg-p_reward[idx])
        print(len(R1))
        RT1, RT2, TT1, TT2 = [], [], [], []
        for i in range(N):
            for j in range(num):
                idx = i*num+j
                if j+1<self.k_len:
                    pad1 = np.zeros(self.k_len-j-1)
                    tmp1, tmp2, tmp3, tmp4 = np.array(R1[i*num:idx+1]), np.array(R2[i*num:idx+1]), np.array(tt1[i*num:idx+1]), np.array(tt2[i*num:idx+1])
                    tmp1, tmp2, tmp3, tmp4 = np.concatenate([pad1, tmp1], axis=0),np.concatenate([pad1, tmp2], axis=0), np.concatenate([pad1, tmp3], axis=0), np.concatenate([pad1, tmp4], axis=0)
                else:
                    tmp1, tmp2, tmp3, tmp4 = np.array(R1[idx-self.k_len+1:idx+1]), np.array(R2[idx-self.k_len+1:idx+1]), np.array(tt1[idx-self.k_len+1:idx+1]), np.array(tt2[idx-self.k_len+1:idx+1])
                
                RT1.append(tmp1)
                RT2.append(tmp2)
                TT1.append(tmp3)
                TT2.append(tmp4)
        cur = [np.array(_state3[0]), np.array(RT1).reshape(-1, self.k_len, 1), np.array(_state3[1]), np.array(mask), np.array(TT1)]
        nex = [np.array(_next_state3[0]), np.array(RT2).reshape(-1, self.k_len, 1), np.array(_next_state3[1]), np.array(mask), np.array(TT2)] 

        return [cur, action, nex, p_reward]

    def select_part_data(self, well_memory):
        state, action, nstate, reward = well_memory
#         rng = np.random.RandomState(42)
        percent = self.dic_traffic_env_conf["PER"]
        random_index = np.random.permutation(len(action))
        random_index = random_index[:int(len(action) * percent)]
        _state1 = state[0][random_index, :, :]
        _state2 = state[1][random_index, :]
        _action = np.array(action)[random_index]
        _next_state1 = nstate[0][random_index, :, :]
        _next_state2 = nstate[1][random_index, :]
        _reward = np.array(reward)[random_index]
        return [[_state1, _state2] , _action, [_next_state1, _next_state2], _reward]
    
    def select_part_data_DT(self, well_memory):
        state, action, nstate, reward = well_memory
        ts = len(action)
        percent = self.dic_traffic_env_conf["PER"]
        random_index = np.random.permutation(len(action))
        random_index = random_index[:int(len(action) * percent)]
        _state1 = state[0][random_index, :, :, :]
        _state2 = state[1][random_index, :, :]
        _state3 = state[2][random_index, :, :]
        _state4 = state[3][random_index, :, :]
        _state5 = state[4][random_index, :]
        _action = np.array(action)[random_index]
        _next_state1 = nstate[0][random_index, :, :, :]
        _next_state2 = nstate[1][random_index, :, :]
        _next_state3 = nstate[2][random_index, :, :]
        _next_state4 = nstate[3][random_index, :, :]
        _next_state5 = nstate[4][random_index, :]
        _reward = np.array(reward)[random_index]
        return [_state1, _state2, _state3, _state4, _state5] , _action, [_next_state1, _next_state2, _next_state3, _next_state4, _next_state5], _reward
    def initialize(self):
        path_check(self.dic_path)
        copy_conf_file(self.dic_path, self.dic_agent_conf, self.dic_traffic_env_conf)
        copy_cityflow_file(self.dic_path, self.dic_traffic_env_conf)

    def run(self):
        replay_memory = self.load_data()
        if self.dic_traffic_env_conf["MODEL_NAME"] in ["DT"]:
            print("=================== prepare samples for DT ==================")
            well_memory = self.prepare_samples_DT(replay_memory, self.t_num)
            well_memory = self.select_part_data_DT(well_memory)
        else:
            well_memory = self.prepare_samples(replay_memory)
            well_memory = self.select_part_data(well_memory)
        replay_memory = None
        
        for cnt_round in range(self.dic_traffic_env_conf["NUM_ROUNDS"]):
            print("round %d starts" % cnt_round)
            round_start_time = time.time()

            print("=============== update model =============")
            generator_start_time = time.time()
            for cnt_gen in range(self.dic_traffic_env_conf["NUM_GENERATORS"]):
                generator_wrapper(cnt_round=cnt_round,
                                  cnt_gen=cnt_gen,
                                  dic_path=self.dic_path,
                                  dic_agent_conf=self.dic_agent_conf,
                                  dic_traffic_env_conf=self.dic_traffic_env_conf,
                                  memory=well_memory)
            generator_end_time = time.time()
            generator_total_time = generator_end_time - generator_start_time


            print("==============  test evaluation =============")
            test_evaluation_start_time = time.time()
            if cnt_round + 10 >= self.dic_traffic_env_conf["NUM_ROUNDS"]:
               
                model_test.test(self.dic_path["PATH_TO_MODEL"], cnt_round,
                                self.dic_traffic_env_conf["RUN_COUNTS"], self.dic_traffic_env_conf)

            test_evaluation_end_time = time.time()
            test_evaluation_total_time = test_evaluation_end_time - test_evaluation_start_time


            print("Generator time: ", generator_total_time)
            print("test_evaluation time:", test_evaluation_total_time)

            print("round {0} ends, total_time: {1}".format(cnt_round, time.time()-round_start_time))

    def prepare_samples(self, memory):
        """
        [state, action, next_state, final_reward, average_reward, vs, vs8, vs200, vs2008]
        """
        state, action, next_state, p_reward, ql_reward, vs, vs8, vs200, vs2008  = memory
        used_feature = self.dic_traffic_env_conf["LIST_STATE_FEATURE"]
        print(used_feature)
        memory_size = len(action)
        _state = [[], None]
        _next_state = [[], None]
        for feat_name in used_feature:
            if feat_name == "phase12":
                _state[1] = np.array(state[feat_name])
                _next_state[1] = np.array(next_state[feat_name])
            else:
                _state[0].append(np.array(state[feat_name]).reshape(memory_size, 12, -1))
                _next_state[0].append(np.array(next_state[feat_name]).reshape(memory_size, 12, -1))
        # ========= generate reaward information ===============
        if "pressure" in self.dic_traffic_env_conf["DIC_REWARD_INFO"].keys():
            my_reward = p_reward
        elif "queue_length" in self.dic_traffic_env_conf["DIC_REWARD_INFO"].keys() :
            my_reward = ql_reward
        elif "vs" in self.dic_traffic_env_conf["DIC_REWARD_INFO"].keys() :
            my_reward = vs
        elif "vs8" in self.dic_traffic_env_conf["DIC_REWARD_INFO"].keys() :
            my_reward = vs8
        elif "vs200" in self.dic_traffic_env_conf["DIC_REWARD_INFO"].keys() :
            my_reward = vs200
        elif "vs2008" in self.dic_traffic_env_conf["DIC_REWARD_INFO"].keys() :
            my_reward = vs2008
        
        return [[np.concatenate(_state[0], axis=-1), _state[1]], action, [np.concatenate(_next_state[0], axis=-1), _next_state[1]], my_reward]