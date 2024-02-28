"""
Ablation study: without ER
"""
from numba import jit
from tensorflow.keras.layers import Input, Dense, Reshape,  Lambda,  Activation, Embedding, MultiHeadAttention, Subtract, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from .network_agent import NetworkAgent
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
import copy
from tensorflow.keras.models import model_from_json, load_model
import os

    
class DataLightAgent3(NetworkAgent):
    def build_network(self):
        ins0 = Input(shape=(12, self.num_feat), name="input_total_features")
        ins1 = Input(shape=(12, ), name="input_cur_phase")
        uni_dim = 6
        cur_phase_emb = Activation('sigmoid')(Embedding(2, uni_dim, input_length=12)(ins1))
        cur_phase_emb = Reshape((12, 1, uni_dim))(cur_phase_emb)
        ins01 = Reshape((12, 1, self.num_feat))(ins0)
        dds = [Dense(uni_dim, activation="sigmoid") for _ in range(self.num_feat)]
        feats = tf.split(ins01, self.num_feat, axis=3)
        feat_embs = []
        for i in range(self.num_feat):
            feat_embs.append(dds[i](feats[i]))
        feats2 = []
        num_seg = int(self.num_feat/3)
        for i in range(num_seg):
            tmp = tf.concat([feat_embs[i], feat_embs[i+num_seg], feat_embs[i+num_seg*2]], axis=2)
            feats2.append(Dense(16, activation='relu')(Reshape((12, 1, 3*uni_dim))(tmp)))
        feat_emb = tf.concat(feats2, axis=2)
        feat_emb = PositionalEncodingLayer(num_seg, 16)(feat_emb)
        feat_emb = MultiHeadAttention(4, 16, attention_axes=2)(feat_emb, feat_emb)
        feat_emb = Reshape((12,-1))(feat_emb)
        cur_phase_emb = Reshape((12, uni_dim))(cur_phase_emb)
        feat_emb = tf.concat([feat_emb, cur_phase_emb], axis=2)
        feat_emb = Dense(24, activation="relu")(feat_emb)
        lane_feat_s = tf.split(feat_emb, 12, axis=1)
        MHA1 = MultiHeadAttention(4, 24, attention_axes=1)
        Mean1 = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))
        phase_feats_map_2 = []
        for i in range(self.num_phases):
            tmp_feat_1 = tf.concat([lane_feat_s[idx] for idx in self.phase_map[i]], axis=1)
            tmp_feat_2 = MHA1(tmp_feat_1, tmp_feat_1)
            tmp_feat_3 = Mean1(tmp_feat_2)
            phase_feats_map_2.append(tmp_feat_3)
        phase_feat_all = tf.concat(phase_feats_map_2, axis=1)
        att_encoding = MultiHeadAttention(4, 24, attention_axes=1)(phase_feat_all, phase_feat_all) #24
        hidden = Dense(20, activation="relu")(att_encoding)
        hidden = Dense(20, activation="relu")(hidden)
        q_values = self.dueling_block(hidden)
        network = Model(inputs=[ins0, ins1],
                        outputs=q_values)
        
        network.compile()
        network.summary()
        return network
    
    def dueling_block(self, inputs):
        tmp_v = Dense(20, activation="relu", name="dense_values")(inputs)
        tmp_v = Reshape((80,))(tmp_v)
        value = Dense(1, activation="linear", name="dueling_values")(tmp_v)
        tmp_a = Dense(20, activation="relu", name="dense_a")(inputs)
        a = Dense(1, activation="linear", name="dueling_advantages")(tmp_a)
        a = Reshape((4,))(a)
        means = Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(a)
        advantages = Subtract()([a, means])
        q_values = Add(name='dueling_q_values')([value, advantages])
        return q_values
    
    def choose_action(self, states):
        dic_state_feature_arrays = {}
        cur_phase_info = []
        used_feature = copy.deepcopy(self.dic_traffic_env_conf["LIST_STATE_FEATURE"])
        # print(used_feature)
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
    
    def epsilon_choice(self, q_values):
        max_1 = np.expand_dims(np.argmax(q_values, axis=-1), axis=-1)
        rand_1 = np.random.randint(4, size=(len(q_values), 1))
        _p = np.concatenate([max_1, rand_1], axis=-1)
        select = np.random.choice([0, 1], size=len(q_values), p=[1 - self.dic_agent_conf["EPSILON"],
                                                                 self.dic_agent_conf["EPSILON"]])
        act = _p[np.arange(len(q_values)), select]
        return act
    def build_network_from_copy(self, network_copy):
        """Initialize a Q network from a copy"""
        network_structure = network_copy.to_json()
        network_weights = network_copy.get_weights()
        network = model_from_json(network_structure, custom_objects={'PositionalEncodingLayer': PositionalEncodingLayer})
        network.set_weights(network_weights)
        network.compile()
        return network
    def load_network(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network = load_model(os.path.join(file_path, "%s.h5" % file_name), custom_objects={'PositionalEncodingLayer': PositionalEncodingLayer})
        print("succeed in loading model %s" % file_name)
    def load_network_bar(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network_bar = load_model(os.path.join(file_path, "%s.h5" % file_name),custom_objects={'PositionalEncodingLayer': PositionalEncodingLayer})
        print("succeed in loading model %s" % file_name)

    def shuffle_data(self, state, action, nstate, reward):
        ts = len(action)
        random_index = np.random.permutation(ts)
        _state1 = state[0][random_index, :, :]
        _state2 = state[1][random_index, :]
        _action = np.array(action)[random_index]
        _next_state1 = nstate[0][random_index, :, :]
        _next_state2 = nstate[1][random_index, :]
        _reward = np.array(reward)[random_index]
        return [_state1, _state2] , _action, [_next_state1, _next_state2], _reward
    
    # @tf.function(jit_compile=True)
    def train_network(self, well_memory, cnt_round):
        _state, _action, _next_state, _reward = well_memory
        epochs = self.dic_agent_conf["EPOCHS"]
        if cnt_round < 80:
            lr = self.dic_agent_conf["LEARNING_RATE"]
        else:
            lr = self.dic_agent_conf["LEARNING_RATE2"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(_action))
        num_batch = int(np.floor((len(_action) / batch_size)))
        loss_fn = MeanSquaredError()
        optimizer = Adam(lr=lr)

        for epoch in range(epochs):
            # ==== shuffle the samples ============ 
            _state, _action, _next_state, _reward = self.shuffle_data(_state, _action, _next_state, _reward)
            for ba in range(int(num_batch)):
                # prepare batch data
                batch_Xs1 = [_state[0][ba*batch_size:(ba+1)*batch_size, :, :], _state[1][ba*batch_size:(ba+1)*batch_size, :]]
                
                batch_Xs2 = [_next_state[0][ba*batch_size:(ba+1)*batch_size, :, :], _next_state[1][ba*batch_size:(ba+1)*batch_size, :]]
                batch_r = _reward[ba*batch_size:(ba+1)*batch_size]
                batch_a = _action[ba*batch_size:(ba+1)*batch_size]

                with tf.GradientTape() as tape:
                    tape.watch(self.q_network.trainable_weights)
                    # calcualte basic loss
                    tmp_cur_q = self.q_network(batch_Xs1)
                    tmp_next_q = self.q_network_bar(batch_Xs2)
                    tmp_target = np.copy(tmp_cur_q)
                    tmp_target = calculate_target_q(tmp_target, batch_size, batch_r, batch_a, np.array(tmp_next_q), self.dic_agent_conf["NORMAL_FACTOR"], self.dic_agent_conf["GAMMA"])
                    base_loss = tf.reduce_mean(loss_fn(tmp_target, tmp_cur_q))

                    replay_action_one_hot = tf.one_hot(batch_a, 4, 1., 0., name='action_one_hot')
                    replay_chosen_q = tf.reduce_sum(tmp_cur_q * replay_action_one_hot)
                    dataset_expec = tf.reduce_mean(replay_chosen_q)
                    negative_sampling = tf.reduce_mean(tf.reduce_logsumexp(tmp_cur_q, 1))
                    min_q_loss = (negative_sampling - dataset_expec)
                    min_q_loss = min_q_loss * self.min_q_weight
                    tmp_loss = base_loss + min_q_loss
                    
                    grads = tape.gradient(tmp_loss, self.q_network.trainable_weights)
                    optimizer.apply_gradients(zip(grads, self.q_network.trainable_weights))
                print("===== Epoch {} | Batch {} / {} | Loss {}".format(epoch, ba, num_batch, tmp_loss))

@jit(nopython=True)
def calculate_target_q(target, bs, r, a, nextq, norm, gamma):
    for i in range(bs):
        target[i, a[i]] = r[i] / norm + gamma * np.max(nextq[i, :])
    return target


class PositionalEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, num_positions, d_model, **kwargs):
        super(PositionalEncodingLayer, self).__init__(**kwargs)
        self.num_positions = num_positions
        self.d_model = d_model
        self.pos_embedding = self.add_weight(
            "pos_embedding", shape=[1, 1, num_positions, d_model], initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return inputs + self.pos_embedding
    def get_config(self):
        config = super(PositionalEncodingLayer, self).get_config()
        config.update({
            "num_positions": self.num_positions,
            "d_model": self.d_model
        })
        return config










