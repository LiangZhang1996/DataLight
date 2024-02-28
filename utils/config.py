from models.datalight import DataLightAgent
from models.datalight2 import DataLightAgent2
from models.datalight3 import DataLightAgent3
from models.attentionlight import AttentionLightAgent
from models.attentionlightBC import AttentionLightAgentBC
from models.attentionlightCQL import AttentionLightAgentCQL
from models.attentionlightBCQ import AttentionLightAgentBCQ
from models.lightDT import LightAgentDT

DIC_AGENTS = {
    "DataLight": DataLightAgent,
    "DataLight2": DataLightAgent2,
    "DataLight3": DataLightAgent3,
    "AttentionLight": AttentionLightAgent,
    "AttentionLightBC": AttentionLightAgentBC,
    "AttentionLightCQL": AttentionLightAgentCQL,
    "AttentionLightBCQ": AttentionLightAgentBCQ,
    "DT":LightAgentDT,
    
}

DIC_PATH = {
    "PATH_TO_MODEL": "model/default",
    "PATH_TO_WORK_DIRECTORY": "records/default",
    "PATH_TO_DATA": "data/template",
    "PATH_TO_PRETRAIN_MODEL": "model/default",
    "PATH_TO_ERROR": "errors/default",
}

dic_traffic_env_conf = {
    "PER": 1,
    "VMAX": 11.112,
    "MIN_Q_W": 0.0001,

    "MODEL_NAME": None,
    "TOP_K_ADJACENCY": 5,

    "ACTION_PATTERN": "set",
    "NUM_INTERSECTIONS": 1,

    "OBS_LENGTH": 111,
    "MIN_ACTION_TIME": 10,
    "MEASURE_TIME": 10,

    "BINARY_PHASE_EXPANSION": True,
    "K_LEN": 5,

    "YELLOW_TIME": 5,
    "ALL_RED_TIME": 0,
    "NUM_PHASES": 4,
    "NUM_LANES": [3, 3, 3, 3],

    "INTERVAL": 1,

    "PHASE_MAP": [[1, 4], [7, 10], [0, 3], [6, 9]],

    "NUM_LANE": 12,

    "LIST_STATE_FEATURE": [
        "lane_num_vehicle_in",
        "lane_num_vehicle_out",
        "lane_queue_vehicle_in",
        "lane_queue_vehicle_out",
        "traffic_movement_pressure_queue",
        "traffic_movement_pressure_queue_efficient",
        "lane_run_in_part",
        "lane_queue_in_part",
        "num_in_seg",
        "adjacency_matrix",
        "new_phase",
        "phase12"
    ],
    
    "DIC_REWARD_INFO": {
        "queue_length": 0,
        "pressure": 0,
    },
    "PHASE": {
        1: [0, 1, 0, 1, 0, 0, 0, 0],
        2: [0, 0, 0, 0, 0, 1, 0, 1],
        3: [1, 0, 1, 0, 0, 0, 0, 0],
        4: [0, 0, 0, 0, 1, 0, 1, 0]
        },
    "PHASE12": {
        1: [0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1],
        2: [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],
        3: [1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1],
        4: [0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]
        },
    
}

DIC_BASE_AGENT_CONF = {
    "D_DENSE": 20,
    "LEARNING_RATE": 0.001,
    "LEARNING_RATE2": 0.0002,
    "PATIENCE": 10,
    "BATCH_SIZE": 500,
    "EPOCHS": 2,
    "SAMPLE_SIZE": 3000,
    "MAX_MEMORY_LEN": 12000,

    "UPDATE_Q_BAR_FREQ": 5,
    "UPDATE_Q_BAR_EVERY_C_ROUND": False,

    "GAMMA": 0.8,
    "NORMAL_FACTOR": 20,

    "EPSILON": 0.8,
    "EPSILON_DECAY": 0.95,
    "MIN_EPSILON": 0.2,
    "LOSS_FUNCTION": "mean_squared_error",
}
