"""
Model testing for DT
"""
import json
import os
import time
from multiprocessing import Process
from utils import config
from utils.utils import merge
from utils.cityflow_env import CityFlowEnv
import argparse
import shutil
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
multi_process = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-memo",       type=str,               default='benchmark_0105_31_t')
    parser.add_argument("-old_memo",   type=str,               default='benchmark_0105_31')
    parser.add_argument("-model",       type=str,               default="DT") 
    parser.add_argument("-old_dir",    type=str,  default='anon_3_4_jinan_real3.json_01_06_12_37_43')
    parser.add_argument("-old_round",  type=str,                default="round_75")
    parser.add_argument("-workers",     type=int,                default=2)
    parser.add_argument("-hangzhou",    action="store_true",     default=0)
    parser.add_argument("-jinan",       action="store_true",     default=1)
    parser.add_argument("-newyork2", action="store_true",        default=1)
    return parser.parse_args()


def main(args):
    if args.hangzhou:
        count = 3600
        road_net = "4_4"
        traffic_file_list = ["anon_4_4_hangzhou_real.json",  "anon_4_4_hangzhou_real_5816.json"]
        num_rounds = 1
        template = "Hangzhou"
    elif args.jinan:
        count = 3600
        road_net = "3_4"
        traffic_file_list = ["anon_3_4_jinan_real_2000.json",  "anon_3_4_jinan_real_2500.json"]
        num_rounds = 1
        template = "Jinan"
    elif args.newyork2:
        count = 3600
        road_net = "28_7"
        traffic_file_list = ["anon_28_7_newyork_real_double.json", "anon_28_7_newyork_real_triple.json"]
        num_rounds = 1
        template = "newyork_28_7"

    NUM_ROW = int(road_net.split('_')[0])
    NUM_COL = int(road_net.split('_')[1])
    num_intersections = NUM_ROW * NUM_COL
    print('num_intersections:', num_intersections)
    print(traffic_file_list)

    old_memo = args.old_memo
    old_dir = args.old_dir
    old_round = args.old_round
    old_model_path = os.path.join("model", old_memo, old_dir)
    process_list = []
    n_workers = args.workers
    for traffic_file in traffic_file_list:
        dic_agent_conf_extra = {
            "CNN_layers": [[32, 32]],
        }
        deploy_dic_agent_conf = merge(getattr(config, "DIC_BASE_AGENT_CONF"), dic_agent_conf_extra)
        dic_traffic_env_conf_extra = {
            "K_LEN": 2,
            "T_NUM": 360,
            "RTG": -350, 
            "OBS_LENGTH": 111,
            "MIN_ACTION_TIME": 10,
            "MEASURE_TIME": 10,

            "test_rounds": old_round,
            "NUM_ROUNDS": num_rounds,
            "NUM_GENERATORS": 1,
            "NUM_AGENTS": 1,
            "NUM_INTERSECTIONS": num_intersections,
            "RUN_COUNTS": count,
            "MODEL_NAME": args.model,
            "NUM_ROW": NUM_ROW,
            "NUM_COL": NUM_COL,
            "TRAFFIC_FILE": traffic_file,
            "ROADNET_FILE": "roadnet_{0}.json".format(road_net),
            "LIST_STATE_FEATURE": [
                "phase12",
                "traffic_movement_pressure_queue_efficient",
                "lane_run_in_part",
            ],
            "DIC_REWARD_INFO": {
                "pressure": -0.25,
            },
        }

        dic_path = {
            "PATH_TO_MODEL": old_model_path,  # use old model path
            "PATH_TO_WORK_DIRECTORY": os.path.join("records", args.memo, traffic_file + "_" +
                                                   time.strftime('%m_%d_%H_%M_%S', time.localtime(
                                                       time.time()))),
            "PATH_TO_DATA": os.path.join("data", template, str(road_net))
        }

        deploy_dic_traffic_env_conf = merge(config.dic_traffic_env_conf, dic_traffic_env_conf_extra)

        multi_process = True
        if multi_process:
            tsr = Process(target=testor_wrapper,
                          args=(deploy_dic_agent_conf,
                                deploy_dic_traffic_env_conf,
                                dic_path,
                                old_round))
            process_list.append(tsr)
        else:
            testor_wrapper(deploy_dic_agent_conf,
                           deploy_dic_traffic_env_conf,
                           dic_path,
                           old_round)

    if multi_process:
        for i in range(0, len(process_list), n_workers):
            i_max = min(len(process_list), i + n_workers)
            for j in range(i, i_max):
                print(j)
                print("start_traffic")
                process_list[j].start()
                print("after_traffic")
            for k in range(i, i_max):
                print("traffic to join", k)
                process_list[k].join()
                print("traffic finish join", k)

    return args.memo


def testor_wrapper(dic_agent_conf, dic_traffic_env_conf, dic_path, old_round):
    testor = Testor(dic_agent_conf,
                    dic_traffic_env_conf,
                    dic_path,
                    old_round)
    testor.main()
    print("============= restor wrapper end =========")


class Testor:
    def __init__(self, dic_agent_conf, dic_traffic_env_conf, dic_path, old_round):
        self.dic_agent_conf = dic_agent_conf
        self.dic_traffic_env_conf = dic_traffic_env_conf
        self.dic_path = dic_path

        self._path_check()
        self._copy_conf_file()
        self._copy_anon_file()

        agent_name = self.dic_traffic_env_conf["MODEL_NAME"]
        # use one-model
        self.agent = config.DIC_AGENTS[agent_name](
            dic_agent_conf=dic_agent_conf,
            dic_traffic_env_conf=dic_traffic_env_conf,
            dic_path=dic_path,
            cnt_round=0,
            intersection_id=str(0)
        )

        self.path_to_log = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        if not os.path.exists(self.path_to_log):
            os.makedirs(self.path_to_log)

    def main(self): 
        rounds = ["round_" + str(i) for i in range(115, 120)]
        for old_round in rounds:
            self.path_to_log = os.path.join(self.dic_path["PATH_TO_WORK_DIRECTORY"], "test_round", old_round)
            if not os.path.exists(self.path_to_log):
                os.makedirs(self.path_to_log)
            self.env = CityFlowEnv(path_to_log=self.path_to_log,
                                   path_to_work_directory=self.dic_path["PATH_TO_WORK_DIRECTORY"],
                                   dic_traffic_env_conf=self.dic_traffic_env_conf)
            self.agent.load_network("{0}_inter_0".format(old_round))

            self.run()

    def run(self):
        done = False
        step_num = 0
        _states = [[] for _ in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"])]
        _rs = [[] for _ in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"])]
        _tt = [[] for _ in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"])]
        _mask = [[] for _ in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"])]
        rtg = [0 for _ in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"])]
        _rtgs = [[] for _ in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"])]
        k_len = self.dic_traffic_env_conf["K_LEN"]
        my_rtg = self.dic_traffic_env_conf["RTG"]
        
        state = self.env.reset()
        running_start_time = time.time()
        while not done and step_num < int(self.dic_traffic_env_conf["RUN_COUNTS"]/self.dic_traffic_env_conf["MIN_ACTION_TIME"]):
            step_start_time = time.time()
            if step_num < k_len-1:
                tmp_mask = create_mask(k_len, k_len-step_num-1)
            else:
                tmp_mask = create_mask(k_len, 0)
            for i in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
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
            for i in range(self.dic_traffic_env_conf["NUM_INTERSECTIONS"]):
                _states[i] = _states[i][-k_len:]
                _tt[i] = _tt[i][-k_len:]
                _rtgs[i] = _rtgs[i][-k_len:]
                _mask[i] = _mask[i][-1:]
                tmp1, tmp2 = convert_states(_states[i], self.dic_traffic_env_conf["LIST_STATE_FEATURE"])
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
            action_list = self.agent.choose_action(cur)
            next_state, reward = self.env.step(action_list)

            print("time: {0}, running_time: {1}".format(
                self.env.get_current_time() - self.dic_traffic_env_conf["MIN_ACTION_TIME"],
                time.time() - step_start_time))
            state = next_state
            step_num += 1

        running_time = time.time() - running_start_time
        log_start_time = time.time()
        print("=========== start env logging ===========")
        self.env.batch_log_2()
        log_time = time.time() - log_start_time
        # self.env.end_anon()
        print("running_time: ", running_time)
        print("log_time: ", log_time)

    def _path_check(self):
        # check path
        if os.path.exists(self.dic_path["PATH_TO_WORK_DIRECTORY"]):
            if self.dic_path["PATH_TO_WORK_DIRECTORY"] != "records/default":
                raise FileExistsError
            else:
                pass
        else:
            os.makedirs(self.dic_path["PATH_TO_WORK_DIRECTORY"])

    def _copy_conf_file(self, path=None):
        if path is None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        json.dump(self.dic_agent_conf, open(os.path.join(path, "agent.conf"), "w"),
                  indent=4)
        json.dump(self.dic_traffic_env_conf,
                  open(os.path.join(path, "traffic_env.conf"), "w"), indent=4)

    def _copy_anon_file(self, path=None):
        if path is None:
            path = self.dic_path["PATH_TO_WORK_DIRECTORY"]
        shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], self.dic_traffic_env_conf["TRAFFIC_FILE"]),
                        os.path.join(path, self.dic_traffic_env_conf["TRAFFIC_FILE"]))
        shutil.copy(os.path.join(self.dic_path["PATH_TO_DATA"], self.dic_traffic_env_conf["ROADNET_FILE"]),
                    os.path.join(path, self.dic_traffic_env_conf["ROADNET_FILE"]))

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

if __name__ == "__main__":
    args = parse_args()
    main(args)
