"""
This model adopt AttentionLight[https://github.com/LiangZhang1996/AttentionLight.git] as its base neural network.
"""

from tensorflow.keras.layers import Input, Dense, Reshape,  Activation, Embedding, add, MultiHeadAttention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from .network_agent import NetworkAgent
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
import copy


class AttentionLightAgentBCQ(NetworkAgent):
    
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
        # ===============   for q ==================
        hidden1 = Dense(20, activation="relu")(hidden)
        phase_feature_final = Dense(1, activation="linear", name="beformerge")(hidden1)
        q_values = Reshape((4,))(phase_feature_final)
        # ================  for i  =================
        hidden2 = Dense(20, activation="relu")(hidden)
        phase_feature_final2 = Dense(1, activation="linear",name="beformerge2")(hidden2)
        i_values = Reshape((4,))(phase_feature_final2)
        i_values2 = tf.keras.activations.softmax(i_values, axis=1)
        network = Model(inputs=[ins0, ins1],
                        outputs=[q_values, i_values2, i_values])
        
        network.compile()
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
        
        q_values, imt, i_values = self.q_network.predict([state_input, np.array(cur_phase_info)])
        
        action = np.argmax(q_values, axis=1)
        return action

    def shuffle_data(self, state, action, nstate, reward):
        percent = self.dic_traffic_env_conf["PER"]
        ts = int(len(action) * percent)
        random_index = np.random.permutation(ts)
        _state1 = state[0][random_index, :, :]
        _state2 = state[1][random_index, :]
        _action = np.array(action)[random_index]
        _next_state1 = nstate[0][random_index, :, :]
        _next_state2 = nstate[1][random_index, :]
        _reward = np.array(reward)[random_index]
        return [_state1, _state2] , _action, [_next_state1, _next_state2], _reward
    
    def train_network(self, well_memory, cnt_round):
        _state, _action, _next_state, _reward = well_memory
        epochs = self.dic_agent_conf["EPOCHS"]
        if cnt_round < 80:
            lr = self.dic_agent_conf["LEARNING_RATE"]
        else:
            lr = self.dic_agent_conf["LEARNING_RATE2"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(_action))
        loss_fn = MeanSquaredError()
        optimizer = Adam(lr=lr)
        for epoch in range(epochs):
            _state, _action, _next_state, _reward = self.shuffle_data(_state, _action, _next_state, _reward)
            num_batch = int(np.floor((len(_action) / batch_size)))
            for ba in range(int(num_batch)):
                batch_Xs1 = [_state[0][ba*batch_size:(ba+1)*batch_size, :, :], _state[1][ba*batch_size:(ba+1)*batch_size, :]]
                batch_Xs2 = [_next_state[0][ba*batch_size:(ba+1)*batch_size, :, :], _next_state[1][ba*batch_size:(ba+1)*batch_size, :]]
                batch_r = _reward[ba*batch_size:(ba+1)*batch_size]
                batch_a = _action[ba*batch_size:(ba+1)*batch_size]
                with tf.GradientTape() as tape:
                    tape.watch(self.q_network.trainable_weights)
                    tmp_next_q, tmp_next_imt, tmp_next_i = self.q_network_bar(batch_Xs2)
                    tmp_imt = tmp_next_imt
                    tmp_imt = (tmp_imt/K.max(tmp_imt, axis=1, keepdims=True)) > self.threshold
                    tmp_imt = np.array(tmp_imt) * tmp_next_q + (1-np.array(tmp_imt))*1e-8
                    tmp_next_action = np.argmax(tmp_imt, axis=1)
                    tmp_next_q2, tmp_next_imt2, tmp_next_i2 = self.q_network_bar(batch_Xs2)
                    tmp_cur_q, tmp_cur_imt, tmp_cur_i = self.q_network(batch_Xs1)
                    tmp_target = np.copy(tmp_cur_q)
                    for i in range(batch_size):
                        tmp_target[i, batch_a[i]] = batch_r[i] / self.dic_agent_conf["NORMAL_FACTOR"] + \
                                                    self.dic_agent_conf["GAMMA"] * \
                                                   tmp_next_q2[i, tmp_next_action[i]]
                        
                    q_loss = tf.keras.losses.Huber()(tmp_target, tmp_cur_q)
                    i_loss = tf.keras.losses.CategoricalCrossentropy()(tmp_cur_imt, np.eye(4)[batch_a])
                    i_loss = tf.cast(i_loss, dtype='float32')
                    q_loss = tf.cast(q_loss, dtype="float32")
                    i_constrain = 1e-2*np.mean(np.power(tmp_cur_i, 2), dtype='float32')
                    tmp_loss = q_loss + i_loss*0.001 + i_constrain*0.01
                    grads = tape.gradient(tmp_loss, self.q_network.trainable_weights)
                    optimizer.apply_gradients(zip(grads, self.q_network.trainable_weights))
                print("===== Epoch {} | Batch {} / {} | Loss {}".format(epoch, ba, num_batch, tmp_loss))













