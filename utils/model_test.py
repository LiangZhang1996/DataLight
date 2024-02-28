from .config import DIC_AGENTS
from copy import deepcopy
from .cityflow_env import CityFlowEnv
import json
import os
import numpy as np


def test(model_dir, cnt_round, run_cnt, _dic_traffic_env_conf):
    dic_traffic_env_conf = deepcopy(_dic_traffic_env_conf)
    records_dir = model_dir.replace("model", "records")
    model_round = "round_%d" % cnt_round
    dic_path = {"PATH_TO_MODEL": model_dir, "PATH_TO_WORK_DIRECTORY": records_dir}
    with open(os.path.join(records_dir, "agent.conf"), "r") as f:
        dic_agent_conf = json.load(f)
    if os.path.exists(os.path.join(records_dir, "anon_env.conf")):
        with open(os.path.join(records_dir, "anon_env.conf"), "r") as f:
            dic_traffic_env_conf = json.load(f)
    dic_traffic_env_conf["RUN_COUNTS"] = run_cnt

    agents = []
    for i in range(dic_traffic_env_conf['NUM_AGENTS']):
        agent_name = dic_traffic_env_conf["MODEL_NAME"]
        agent = DIC_AGENTS[agent_name](
            dic_agent_conf=dic_agent_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
            cnt_round=0,
            intersection_id=str(i)
        )
        agents.append(agent)
    
    print("========== start testing model ===========")
    for i in range(dic_traffic_env_conf['NUM_AGENTS']):
        agents[i].load_network("{0}_inter_{1}".format(model_round, agents[i].intersection_id))
        
    if dic_traffic_env_conf["MODEL_NAME"] in ["DT"]:
        _states = [[] for _ in range(dic_traffic_env_conf["NUM_INTERSECTIONS"])]
        _rs = [[] for _ in range(dic_traffic_env_conf["NUM_INTERSECTIONS"])]
        _tt = [[] for _ in range(dic_traffic_env_conf["NUM_INTERSECTIONS"])]
        _mask = [[] for _ in range(dic_traffic_env_conf["NUM_INTERSECTIONS"])]
        rtg = [0 for _ in range(dic_traffic_env_conf["NUM_INTERSECTIONS"])]
        _rtgs = [[] for _ in range(dic_traffic_env_conf["NUM_INTERSECTIONS"])]
            
    path_to_log = os.path.join(dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", model_round)
    if not os.path.exists(path_to_log):
        os.makedirs(path_to_log)
    env = CityFlowEnv(
            path_to_log=path_to_log,
            path_to_work_directory=dic_path["PATH_TO_WORK_DIRECTORY"],
            dic_traffic_env_conf=dic_traffic_env_conf
        )

    step_num = 0
    k_len = dic_traffic_env_conf["K_LEN"]
    my_rtg = dic_traffic_env_conf["RTG"]
    total_time = dic_traffic_env_conf["RUN_COUNTS"]
    state = env.reset()
    while step_num < int(total_time / dic_traffic_env_conf["MIN_ACTION_TIME"]):
            
        if dic_traffic_env_conf["MODEL_NAME"] in ["DT"]:
            action_list = []
            if step_num < k_len-1:
                tmp_mask = create_mask(k_len, k_len-step_num-1)
            else:
                tmp_mask = create_mask(k_len, 0)
            for i in range(dic_traffic_env_conf["NUM_INTERSECTIONS"]):
                # states
                _states[i].append(state[i])
                # reward
                if step_num > 0:
                    _rs[i].append(reward[i])
                # rtgs
                if step_num == 0:
                    rtg[i] = my_rtg
                else:
                    rtg[i] = rtg[i] - _rs[i][-1]
                _rtgs[i].append(rtg[i])
                    # time step
                _tt[i].append(step_num)
                # mask
                _mask[i].append(tmp_mask)
                
            new_states = [[] for _ in range(5)]
            for i in range(dic_traffic_env_conf["NUM_INTERSECTIONS"]):
                _states[i] = _states[i][-k_len:]
                _tt[i] = _tt[i][-k_len:]
                _rtgs[i] = _rtgs[i][-k_len:]
                _mask[i] = _mask[i][-1:]
                    
                tmp1, tmp2 = convert_states(_states[i], dic_traffic_env_conf["LIST_STATE_FEATURE"])

                tmp3, tmp4 = np.array(_rtgs[i]), np.array(_tt[i])
                if step_num < k_len-1:
                    pad1, pad2 = np.zeros((k_len-step_num-1,12, 2)), np.zeros(( k_len-step_num-1,12))
                    tmp1, tmp2 = np.concatenate([pad1, tmp1], axis=0), np.concatenate([pad2, tmp2], axis=0)
                    pad3 = np.zeros(k_len-step_num-1)
                    tmp3, tmp4 = np.concatenate([pad3, tmp3], axis=0), np.concatenate([pad3, tmp4], axis=0)
                
                new_states[0].append(tmp1)
                new_states[1].append(tmp3)
                new_states[2].append(tmp2)
                new_states[3].append(_mask[i][-1])
                new_states[4].append(tmp4)
                
            cur = [np.array(new_states[0]), np.array(new_states[1]).reshape(-1, k_len, 1), np.array(new_states[2]), np.array(new_states[3]), np.array(new_states[4])]
            action_list = agents[0].choose_action(cur)
                
        else:
            action_list = agents[0].choose_action(state)
            
        next_state, reward = env.step(action_list)
        state = next_state
        step_num += 1

    env.batch_log_2()
    env.end_cityflow()


def convert_states(states, used_feature):
        dic_state_feature_arrays = {}
        cur_phase_info = []
        for feature_name in used_feature:
            dic_state_feature_arrays[feature_name] = []
        for s in states:
            for feature_name in used_feature:
                if feature_name == "phase12":
                    cur_phase_info.append(s[feature_name])
                else:
                    dic_state_feature_arrays[feature_name].append(s[feature_name])
        
        state_input = [np.array(dic_state_feature_arrays[feature_name]).reshape(len(states), 12, -1) for feature_name in
                       used_feature[1:]]
        state_input = np.concatenate(state_input, axis=-1)
        # [batch, 12, dim] -> [1, batch, 12, dim]
        state1, state2 = np.array(state_input), np.array(cur_phase_info)
        return state1, state2
        
def create_mask(seq_length, a):
    total_length = 3 * seq_length
    mask = np.ones((total_length, total_length), dtype=np.float32)
    for i in range(seq_length):
        start_idx = 3 * i
        end_idx = start_idx + 3
        mask[start_idx:end_idx, :end_idx] = 0
    mask[:, :3*a] = 1

    return mask