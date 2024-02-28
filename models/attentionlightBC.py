"""
This model adopt AttentionLight[https://github.com/LiangZhang1996/AttentionLight.git] as its base neural network.
"""

from tensorflow.keras.layers import Input, Dense, Reshape, Activation, Embedding, add, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from .network_agent import NetworkAgent
import numpy as np
import tensorflow as tf
import copy


class AttentionLightAgentBC(NetworkAgent):
    def build_network(self):
        ins0 = Input(shape=(12, self.num_feat), name="input_total_features")
        ins1 = Input(shape=(12, ), name="input_cur_phase")
        ins0s = tf.split(ins0, self.num_feat, 2) 
        cur_phase_emb = Activation('sigmoid')(Embedding(2, 4, input_length=12)(ins1))
        dd = []
        for i in range(self.num_feat):
            dd.append( Dense(4, activation="sigmoid"))
        feats = [] 
        for i in range(self.num_feat):
            feats.append(dd[i](ins0s[i]))
        feats.append(cur_phase_emb)
        feats = tf.concat(feats, axis=2)
        lane_feat_s = tf.split(feats, 12, axis=1)
        lane_embedding = Dense(24, activation="relu")
        phase_feats_2 = []
        for i in range(self.num_phases):
            phase_feats_2.append(add([lane_embedding(lane_feat_s[self.phase_map[i][0]]), lane_embedding(lane_feat_s[self.phase_map[i][1]])]))
        phase_feat_all = tf.concat(phase_feats_2, axis=1)
        att_encoding = MultiHeadAttention(4, 8, attention_axes=1)(phase_feat_all, phase_feat_all)
        hidden = Dense(20, activation="relu")(att_encoding)
        hidden = Dense(20, activation="relu")(hidden)
        hidden = Reshape((20*4,))(hidden)
        phase_feature_final = Dense(20, activation="relu", name="beformerge")(hidden)
        q_values = Dense(4, activation="softmax")(phase_feature_final)
        network = Model(inputs=[ins0, ins1],
                        outputs=q_values)
        network.compile(optimizer=Adam(lr=self.dic_agent_conf["LEARNING_RATE"], epsilon=1e-08),
                        loss="categorical_crossentropy")
        network.summary()
        return network

    def choose_action(self, states):
        dic_state_feature_arrays = {}
        cur_phase_info = []
        used_feature = copy.deepcopy(self.dic_traffic_env_conf["LIST_STATE_FEATURE"])
        for feature_name in used_feature:
            dic_state_feature_arrays[feature_name] = []
        for s in states:
            for feature_name in self.dic_traffic_env_conf["LIST_STATE_FEATURE"]:
                if feature_name == "phase12":
                    cur_phase_info.append(s[feature_name])
                else:
                    dic_state_feature_arrays[feature_name].append(s[feature_name])  
        used_feature.remove("phase12")
        state_input = [np.array(dic_state_feature_arrays[feature_name]).reshape(len(states), 12, -1) for feature_name in
                       used_feature]
        state_input = np.concatenate(state_input, axis=-1)
        q_values = self.q_network.predict([state_input, np.array(cur_phase_info)])
        action = np.argmax(q_values, axis=1)
        return action
    
    def train_network(self, well_memory, cnt_round):
        _state, _action, _next_state, _reward = well_memory
        epochs = self.dic_agent_conf["EPOCHS"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(_action))
        one_hot_action = tf.one_hot(_action, depth=4)
        self.q_network.fit(_state, one_hot_action, batch_size=batch_size, epochs=epochs, shuffle=True,
                           verbose=2, validation_split=0.01)
        
    
                


