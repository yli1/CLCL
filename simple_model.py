import tensorflow as tf
import numpy as np
import random


class SimpleModel(object):
    def __init__(self, args):
        self.args = args
        if args.random_random:
            tf.set_random_seed(random.randint(2, 1000))
        else:
            tf.set_random_seed(args.random_seed)
        self.max_input_length = args.input_length
        self.max_output_length = args.output_length

    def get_optimizer(self, scope=None):
        if self.args.decay_steps <= 0:
            learning_rate = self.args.learning_rate
        else:
            global_step = tf.Variable(0, trainable=False)
            starter_learning_rate = self.args.learning_rate
            decay_steps = self.args.decay_steps
            decay_base = 0.96
            learning_rate = tf.train.exponential_decay(
                starter_learning_rate,
                global_step,
                decay_steps,
                decay_base,
                staircase=True)

        if self.args.max_gradient_norm < 0:
            if self.args.decay_steps <= 0:
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate).minimize(self.loss)
            else:
                self.optimizer = tf.train.AdamOptimizer(
                    learning_rate).minimize(self.loss, global_step=global_step)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate)

            params = tf.trainable_variables(scope)
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(
                gradients, self.args.max_gradient_norm)

            if self.args.decay_steps <= 0:
                optimizer_op = optimizer.apply_gradients(
                    zip(clipped_gradients, params))
            else:
                optimizer_op = optimizer.apply_gradients(
                    zip(clipped_gradients, params), global_step=global_step)
        return optimizer, optimizer_op

    def initialize(self, voc_size, act_size):
        self.V = voc_size
        self.U = act_size
        self.regularization_list = []
        self.create_model()

        optimizer, self.optimizer_op = self.get_optimizer()
        if self.args.continual_learning:
            if self.args.continual_all_params:
                scope = None
            else:
                scope = 'word_embeddings'
            _, self.continual_optimizer_op = self.get_optimizer(scope=scope)

        self.reset_optimizer_op = tf.variables_initializer(
            optimizer.variables())

        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        tf.summary.FileWriter('logs/' + self.args.experiment_id + '/model',
                              self.sess.graph)
        self.sess.run(init)

    def ff(self, x, layers, hidden, out, use_bias=True):
        for i in range(layers):
            if i == layers - 1:
                activation = None
                nodes = out
            else:
                activation = tf.nn.relu
                nodes = hidden
            x = tf.layers.dense(x, nodes, activation=activation,
                                use_bias=use_bias)
        return x

    def noise_regularization(self, rep, noise_weight, reg_coe):
        if reg_coe > 0:
            if self.args.use_l1_norm:
                norm = tf.reduce_mean(tf.abs(rep), -1)
            else:
                norm = tf.reduce_mean(rep ** 2, -1)
            masked_norm = norm * self.input_mask_float
            if self.args.sample_wise_content_noise:
                noise_reg = tf.reduce_sum(
                    masked_norm / tf.to_float(self.batch_size))
            else:
                norm_sum = tf.reduce_sum(masked_norm, -1)
                noise_reg_sample = norm_sum / tf.to_float(self.x_len)
                noise_reg = tf.reduce_mean(noise_reg_sample)
            self.regularization_list.append(reg_coe * noise_reg)

        noisy_rep = rep + noise_weight * tf.random_normal(tf.shape(rep))
        return noisy_rep, rep

    def attention_generation(self, x):
        """
        Generate attention on input sequences for each output node.
        :param x: function sequence Tensor (?, input_len, voc_size)
        :return: Tensor (?, output_len, input_len)
        """
        function_concat = tf.reshape(
            x, [-1, self.max_input_length * self.V], name='function_concat')
        with tf.variable_scope("attention_network"):
            score_stack = self.ff(function_concat, 1, 32,
                                  self.max_output_length * self.max_input_length)
        score = tf.reshape(score_stack,
                           [-1, self.max_output_length, self.max_input_length],
                           name='score')
        with tf.variable_scope("attention_softmax"):
            attention = tf.nn.softmax(score, dim=-1, name='attention')
        return attention

    def get_representations(self):
        x_one_hot = tf.one_hot(self.x, self.V, name='x_one_hot')

        # compute switch
        with tf.variable_scope("compute_switch"):
            switch_score = self.ff(x_one_hot, 1, 32, 1)
            self.switch = tf.nn.sigmoid(
                switch_score / self.args.switch_temperature)
        if self.args.remove_switch:
            primitive = x_one_hot
            function = x_one_hot
        else:
            if self.args.relu_switch:
                switch_primitive = tf.nn.relu(2 * self.switch - 1)
                switch_function = tf.nn.relu(1 - 2 * self.switch)
            else:
                switch_primitive = self.switch
                switch_function = 1 - self.switch
            primitive = tf.multiply(switch_primitive, x_one_hot,
                                    name='primitive')
            function = tf.multiply(switch_function, x_one_hot, name='function')

        # embedding
        if self.args.use_embedding:
            embedding_size = self.args.embedding_size
            primitive = tf.layers.dense(primitive, embedding_size)
            function = tf.layers.dense(function, embedding_size)

        return primitive, function

    def create_model(self):
        self.x = tf.placeholder(tf.int64, shape=(None, self.max_input_length,),
                                name='x')
        self.y = tf.placeholder(tf.int64,
                                shape=(None, self.max_output_length,),
                                name='y')
        self.x_len = tf.placeholder(tf.int32, shape=(None,), name='x_len')
        self.y_len = tf.placeholder(tf.int32, shape=(None,), name='y_len')
        self.noise_weight = tf.placeholder(tf.float32, shape=(),
                                           name='noise_weight')
        self.batch_size = tf.shape(self.x_len)

        # masks
        self.target_mask_float = tf.sequence_mask(
            self.y_len, maxlen=self.max_output_length, dtype=tf.float32)
        self.target_mask_int64 = tf.sequence_mask(
            self.y_len, maxlen=self.max_output_length, dtype=tf.int64)
        self.input_mask_float = tf.sequence_mask(
            self.x_len, maxlen=self.max_input_length, dtype=tf.float32)

        with tf.variable_scope("word_embeddings"):
            primitive, function = self.get_representations()
            if self.args.single_representation:
                function = primitive

        with tf.variable_scope("noise_regularization"):
            if self.args.content_noise:
                primitive, _ = self.noise_regularization(
                    primitive, self.noise_weight, self.args.content_noise_coe)

            if self.args.function_noise:
                function, _ = self.noise_regularization(
                    function, self.noise_weight, self.args.content_noise_coe)

        with tf.variable_scope("generate_attention"):
            self.attention = self.attention_generation(function)

        with tf.variable_scope("prediction"):
            h = tf.matmul(self.attention, primitive)
            with tf.variable_scope("prediction_network"):
                l = self.ff(h, 1, 32, self.U,
                            use_bias=(not self.args.remove_prediction_bias))

        with tf.variable_scope("evaluation"):
            # loss
            loss_sum = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.y, logits=l)
            self.loss = tf.reduce_sum(
                loss_sum * self.target_mask_float / tf.to_float(
                    self.batch_size))

            # switch regularization
            if self.args.reg_coe > 0:
                if self.args.use_entropy_reg:
                    entropy = -(self.switch * tf.log(self.switch) + (
                            1 - self.switch) * tf.log(1 - self.switch))
                else:
                    entropy = self.switch * (1 - self.switch)
                ent = tf.squeeze(entropy, axis=-1)
                reg = tf.reduce_sum(
                    ent * self.input_mask_float / tf.to_float(self.batch_size))
                self.loss += self.args.reg_coe * reg

            if self.args.macro_switch_reg_coe > 0:
                ss = tf.squeeze(self.switch, axis=-1) - 0.5
                ss_value = tf.reduce_sum(ss * self.input_mask_float,
                                         -1) / tf.to_float(self.x_len)
                ss_reg = tf.reduce_mean(ss_value ** 2)
                self.loss += self.args.macro_switch_reg_coe * ss_reg

            for reg in self.regularization_list:
                self.loss += reg

            # word accuracy
            self.prediction = tf.argmax(l, -1) * self.target_mask_int64
            word_equality = tf.to_float(tf.equal(self.y, self.prediction))
            valid_word_equality = word_equality * self.target_mask_float
            self.word_accuracy = tf.reduce_mean(tf.reduce_sum(
                valid_word_equality, -1) / (tf.to_float(self.y_len)))

            # sentence accuracy
            sent_equality = tf.reduce_min(word_equality, axis=-1)
            self.sent_accuracy = tf.reduce_mean(sent_equality)

    def select(self, sample_list, batch_size, start):
        length = len(sample_list[0])
        end = min(start + batch_size, length)
        batch = [samples[start:end] for samples in sample_list]
        return batch, end % length

    def random_select(self, sample_list, batch_size, start):
        length = len(sample_list[0])
        indice = np.random.choice(length, size=batch_size)
        result = [[samples[i] for i in indice] for samples in sample_list]
        return result, start

    def train(self, X, Y, X_len, Y_len, continual=False):
        log_step = 100

        if continual:
            optimizer_op = self.continual_optimizer_op
            batch_size = 1
        else:
            optimizer_op = self.optimizer_op
            batch_size = self.args.batch_size
        fetch = [optimizer_op, self.loss, self.word_accuracy,
                 self.sent_accuracy]

        start = 0
        avg_loss, avg_wa, avg_sa = 0, 0, 0
        for i in range(self.args.epochs):
            if self.args.random_batch:
                batch, start = self.random_select([X, Y, X_len, Y_len],
                                                  batch_size, start)
            else:
                if self.args.shuffle_batch and start == 0:
                    c = list(zip(X, Y, X_len, Y_len))
                    random.shuffle(c)
                    X, Y, X_len, Y_len = zip(*c)
                batch, start = self.select([X, Y, X_len, Y_len], batch_size,
                                           start)

            feed = {self.x: batch[0], self.y: batch[1], self.x_len: batch[2],
                    self.y_len: batch[3],
                    self.noise_weight: self.args.noise_weight}
            _, loss, word_acc, sent_acc = self.sess.run(fetch, feed_dict=feed)
            avg_loss += loss
            avg_wa += word_acc
            avg_sa += sent_acc
            if i % log_step == 0 or i == self.args.epochs - 1:
                print(
                    i, avg_loss / log_step, avg_wa / log_step,
                       avg_sa / log_step)
                avg_loss, avg_wa, avg_sa = 0, 0, 0

    def test(self, X, Y, X_len, Y_len, name, noise_weight=0.0):
        fetch = [self.loss, self.word_accuracy, self.sent_accuracy,
                 self.prediction, self.attention, self.switch]
        feed = {self.x: X, self.y: Y, self.x_len: X_len, self.y_len: Y_len,
                self.noise_weight: noise_weight}
        loss, word_scc, sent_acc, prediction, attention, switch = self.sess.run(
            fetch, feed_dict=feed)
        print(name, loss, word_scc, sent_acc)
        return prediction, attention, switch


class LSTMModel(SimpleModel):
    def attention_generation(self, x):
        units = 8
        n_stacks = 2

        x = tf.unstack(x, self.max_input_length, 1)

        cell_fw = [tf.nn.rnn_cell.BasicRNNCell(units) for _ in range(n_stacks)]
        cell_bw = [tf.nn.rnn_cell.BasicRNNCell(units) for _ in range(n_stacks)]

        x, _, _ = tf.contrib.rnn.stack_bidirectional_rnn(
            cell_fw, cell_bw, x, dtype=tf.float32)

        outputs = tf.stack(x[:self.max_output_length], 1)
        with tf.variable_scope("attention_network"):
            score = self.ff(outputs, 1, 32,
                            self.max_input_length)
        with tf.variable_scope("attention_softmax"):
            attention = tf.nn.softmax(score / 10, dim=-1, name='attention')
        return attention


class S2SModel(SimpleModel):
    def get_decoder_cell(self, encoder_state, encoder_outputs, decoder_cell,
                         num_units, source_sequence_length):
        return decoder_cell, encoder_state

    def get_encoder_bidirectional(self, x):
        encoder_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.args.num_units / 2)
        encoder_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.args.num_units / 2)
        if self.args.use_input_length:
            encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
                encoder_cell_fw, encoder_cell_bw, x,
                sequence_length=self.x_len, dtype=tf.float32)
        else:
            encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
                encoder_cell_fw, encoder_cell_bw, x, dtype=tf.float32)
        h = tf.concat([encoder_state[0].h, encoder_state[1].h], 1)
        c = tf.concat([encoder_state[0].c, encoder_state[1].c], 1)
        state = tf.nn.rnn_cell.LSTMStateTuple(h, c)
        return tf.concat(encoder_outputs, 2), state

    def get_encoder(self, x):
        encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(self.args.num_units)
        if self.args.use_input_length:
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                encoder_cell, x, sequence_length=self.x_len, dtype=tf.float32)
        else:
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                encoder_cell, x, dtype=tf.float32)
        return encoder_outputs, encoder_state

    def get_decoder_input(self):
        if not self.args.use_decoder_input:
            zeros = tf.zeros(shape=self.batch_size)
            unsqueezed = tf.reshape(zeros, [-1, 1, 1])
            decoder_emb_inp = tf.tile(unsqueezed,
                                      [1, self.max_output_length, 1])
            return decoder_emb_inp
        else:
            embeddings = tf.Variable(
                tf.random_uniform([self.U, self.args.output_embedding_size],
                                  -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, self.y)
            decoder_emb_inp = tf.manip.roll(embed, 1, 1)
            return decoder_emb_inp

    def attention_generation(self, x):
        num_units = self.args.num_units
        source_sequence_length = self.x_len

        # Encoder
        if self.args.bidirectional_encoder:
            encoder_outputs, encoder_state = self.get_encoder_bidirectional(x)
        else:
            encoder_outputs, encoder_state = self.get_encoder(x)

        # Decoder
        decoder_emb_inp = self.get_decoder_input()
        decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
        decoder_cell, encoder_state = self.get_decoder_cell(
            encoder_state, encoder_outputs, decoder_cell,
            num_units, source_sequence_length)

        ones = tf.ones(shape=self.batch_size, dtype=tf.int32)
        lengths = ones * self.max_output_length
        helper = tf.contrib.seq2seq.TrainingHelper(decoder_emb_inp, lengths)
        decoder = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper,
                                                  encoder_state)
        outputs, final_state, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(
            decoder)
        logits = outputs.rnn_output

        with tf.variable_scope("attention_network"):
            score = self.ff(logits, 1, 32,
                            self.max_input_length)
        with tf.variable_scope("attention_softmax"):
            if self.args.masked_attention:
                score = score / self.args.attention_temperature
                ex = tf.exp(score) * tf.expand_dims(self.input_mask_float, 1)
                attention = ex / tf.reduce_sum(ex, -1, keepdims=True)
            else:
                attention = tf.nn.softmax(
                    score / self.args.attention_temperature, dim=-1,
                    name='attention')
        return attention


class S2SAttentionModel(S2SModel):
    def get_decoder_cell(self, encoder_state, encoder_outputs, decoder_cell,
                         num_units, source_sequence_length):
        attention_states = encoder_outputs
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(
            num_units, attention_states,
            memory_sequence_length=source_sequence_length)
        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            decoder_cell, attention_mechanism,
            attention_layer_size=num_units)

        initial_state = decoder_cell.zero_state(dtype=tf.float32,
                                                batch_size=self.batch_size)
        initial_state = initial_state.clone(cell_state=encoder_state)

        return decoder_cell, initial_state
