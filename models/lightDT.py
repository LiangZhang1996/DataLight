"""
Application of DT
"""

from tensorflow.keras.layers import Input, Dense, Reshape, MultiHeadAttention, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from .network_agent import NetworkAgent
import numpy as np
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.models import model_from_json, load_model
import os


class LightAgentDT(NetworkAgent):

    def build_network(self):
        ins0 = Input(shape=(self.k_len, 12, self.num_feat), name="input_states")
        ins1 = Input(shape=(self.k_len, 1), name="input_returns_to_go")
        ins2 = Input(shape=(self.k_len, 12, ), name="input_actions")
        ins3 = Input(shape=(self.k_len*3, self.k_len*3), name="input_att_mask")
        ins4 = Input(shape=(self.k_len), name="timestep")
        # Embedding
        s_emb = Dense(32, activation="relu")( Reshape((self.k_len, 12*self.num_feat))(ins0))
        s_emb = Dense(32, activation="relu")(s_emb)
        a_emb = Dense(32, activation="relu")(ins2)
        a_emb = Dense(32, activation="relu")(a_emb)
        r_emb = Dense(32, activation="relu")(ins1)
        r_emb = Dense(32, activation="relu")(r_emb)
        # positional embedding
        pos_emb = LearnedPositionalEmbedding(self.t_num+1, 32)(ins4)
        s_emb = s_emb + pos_emb
        a_emb = a_emb + pos_emb
        r_emb = r_emb + pos_emb
        
        s_emb = Dense(32, activation="relu")(s_emb)
        a_emb = Dense(32, activation="relu")(a_emb)
        r_emb = Dense(32, activation="relu")(r_emb)
        # stack
        s_embs, a_embs, r_embs = tf.split(s_emb, self.k_len, axis=1),tf.split(a_emb, self.k_len, axis=1) ,tf.split(r_emb, self.k_len, axis=1)
        in_embs = []
        for i in range(self.k_len):
            in_embs.extend([r_embs[i], s_embs[i], a_embs[i]])
        in_embs = tf.concat(in_embs, axis=1)
        
        # hiddeen states
        hidden_s = MultiHeadAttention(2, 20, attention_axes=1)(in_embs, in_embs, attention_mask=ins3)
        a_hidden = hidden_s[:, -2: , :]
        a_hidden = Flatten()(a_hidden)
        
        hidden = Dense(20, activation="relu")(a_hidden)
        hidden = Dense(20, activation="relu")(hidden)

        q_values = Dense(4, activation="linear")(hidden)

        network = Model(inputs=[ins0, ins1, ins2, ins3, ins4],
                        outputs=q_values)
        network.compile()
        network.summary()
        return network
    
    def choose_action(self, states):
        q_values = self.q_network.predict(states)
        action = np.argmax(q_values, axis=1)
        return action

    def shuffle_data(self, state, action, nstate, reward):
        ts = len(action)
        random_index = np.random.permutation(ts)
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
    
    def train_network(self, well_memory, cnt_round):
        _state, _action, _next_state, _reward, = well_memory
        epochs = self.dic_agent_conf["EPOCHS"]
        batch_size = min(self.dic_agent_conf["BATCH_SIZE"], len(_action))
        num_batch = int(np.floor((len(_action) / batch_size)))
        loss_fn = MeanSquaredError()
        optimizer = Adam(lr=self.dic_agent_conf["LEARNING_RATE"])
        for epoch in range(epochs):
            _state, _action, _next_state, _reward = self.shuffle_data(_state, _action, _next_state, _reward)
            for ba in range(int(num_batch)):
                # prepare batch data
                batch_Xs1 = [_state[0][ba*batch_size:(ba+1)*batch_size, :, :, :], _state[1][ba*batch_size:(ba+1)*batch_size, :, :],  _state[2][ba*batch_size:(ba+1)*batch_size, :, :],  _state[3][ba*batch_size:(ba+1)*batch_size, :, :],  _state[4][ba*batch_size:(ba+1)*batch_size, :]]
                
                batch_Xs2 = [_next_state[0][ba*batch_size:(ba+1)*batch_size, :, :, :], _next_state[1][ba*batch_size:(ba+1)*batch_size, :, :], _next_state[2][ba*batch_size:(ba+1)*batch_size, :, :],_next_state[3][ba*batch_size:(ba+1)*batch_size, :, :],_next_state[4][ba*batch_size:(ba+1)*batch_size, :]]
                batch_r = _reward[ba*batch_size:(ba+1)*batch_size]
                batch_a = _action[ba*batch_size:(ba+1)*batch_size]

                with tf.GradientTape() as tape:
                    tape.watch(self.q_network.trainable_weights)
                    # calcualte basic loss
                    tmp_cur_q = self.q_network(batch_Xs1)
                    tmp_next_q = self.q_network_bar(batch_Xs2)
                    tmp_target = np.copy(tmp_cur_q)
                    for i in range(batch_size):
                        tmp_target[i, batch_a[i]] = batch_r[i] / self.dic_agent_conf["NORMAL_FACTOR"] + \
                                                    self.dic_agent_conf["GAMMA"] * \
                                                    np.max(tmp_next_q[i, :])
                    base_loss = tf.reduce_mean(loss_fn(tmp_target, tmp_cur_q))
                    tmp_loss = base_loss 
                    grads = tape.gradient(tmp_loss, self.q_network.trainable_weights)
                    optimizer.apply_gradients(zip(grads, self.q_network.trainable_weights))
                print("===== Epoch {} | Batch {} / {} | Loss {}".format(epoch, ba, num_batch, tmp_loss))

    def build_network_from_copy(self, network_copy):
        """Initialize a Q network from a copy"""
        network_structure = network_copy.to_json()
        network_weights = network_copy.get_weights()
        network = model_from_json(network_structure, custom_objects={'LearnedPositionalEmbedding': LearnedPositionalEmbedding})
        network.set_weights(network_weights)
        network.compile()
        return network
    def load_network(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network = load_model(os.path.join(file_path, "%s.h5" % file_name), custom_objects={'LearnedPositionalEmbedding': LearnedPositionalEmbedding})
        print("succeed in loading model %s" % file_name)
    def load_network_bar(self, file_name, file_path=None):
        if file_path is None:
            file_path = self.dic_path["PATH_TO_MODEL"]
        self.q_network_bar = load_model(os.path.join(file_path, "%s.h5" % file_name),custom_objects={'LearnedPositionalEmbedding': LearnedPositionalEmbedding})
        print("succeed in loading model %s" % file_name)

class LearnedPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_timesteps, embedding_dim, **kwargs):
        super(LearnedPositionalEmbedding, self).__init__(**kwargs)
        self.num_timesteps = num_timesteps
        self.embedding_dim = embedding_dim
        self.embedding = tf.keras.layers.Embedding(input_dim=num_timesteps, output_dim=embedding_dim)

    def call(self, timesteps):
        return self.embedding(timesteps)

    def get_config(self):
        config = super(LearnedPositionalEmbedding, self).get_config()
        config.update({
            "num_timesteps": self.num_timesteps,
            "embedding_dim": self.embedding_dim
        })
        return config

