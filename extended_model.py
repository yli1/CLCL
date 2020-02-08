from simple_model import S2SAttentionModel

import random
import tensorflow as tf


class RandRegModel(S2SAttentionModel):
    def __init__(self, args):
        self.args = args
        if args.random_random:
            tf.set_random_seed(random.randint(2, 1000))
        else:
            tf.set_random_seed(args.random_seed)
        self.max_input_length = args.input_length
        self.original_max_output_length = args.output_length

    def get_embeddings(self, inputs, embedding_size):
        embeddings = tf.Variable(
            tf.random_uniform([self.V, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, inputs)
        return embed

    def get_representations(self):
        # embedding
        if self.args.use_embedding:
            primitive = self.get_embeddings(self.x, self.args.embedding_size)
            function = self.get_embeddings(self.x, self.args.embedding_size)

        # compute switch
        with tf.variable_scope("compute_switch"):
            switch_score = self.get_embeddings(self.x, 1)
            self.switch = tf.nn.sigmoid(
                switch_score / self.args.switch_temperature)
        if not self.args.remove_switch:
            if self.args.relu_switch:
                switch_primitive = tf.nn.relu(2 * self.switch - 1)
                switch_function = tf.nn.relu(1 - 2 * self.switch)
            else:
                switch_primitive = self.switch
                switch_function = 1 - self.switch
            primitive = tf.multiply(switch_primitive, primitive,
                                    name='primitive')
            function = tf.multiply(switch_function, function, name='function')

        return primitive, function

    def create_placeholders(self):
        self.x = tf.placeholder(tf.int64, shape=(None, self.max_input_length,),
                                name='x')
        self.y = tf.placeholder(tf.int64,
                                shape=(None, self.original_max_output_length,),
                                name='y')
        self.x_len = tf.placeholder(tf.int32, shape=(None,), name='x_len')
        self.y_len = tf.placeholder(tf.int32, shape=(None,), name='y_len')
        self.noise_weight = tf.placeholder(tf.float32, shape=(),
                                           name='noise_weight')
        self.valid_outputs = tf.placeholder(tf.int32, shape=(),
                                            name='valid_outputs')
        self.batch_size = tf.shape(self.x_len)

        self.max_output_length = tf.reduce_max(self.y_len,
                                               name='max_output_length')
        sliced_y = tf.slice(self.y, [0, 0], [-1, self.max_output_length])
        return sliced_y

    def create_model(self):
        sliced_y = self.create_placeholders()

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
                labels=sliced_y, logits=l)
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
            word_equality = tf.to_float(tf.equal(sliced_y, self.prediction))
            valid_word_equality = word_equality * self.target_mask_float
            self.word_accuracy = tf.reduce_mean(tf.reduce_sum(
                valid_word_equality, -1) / (tf.to_float(self.y_len)))

            # sentence accuracy
            sent_equality = tf.reduce_min(word_equality, axis=-1)
            self.sent_accuracy = tf.reduce_mean(sent_equality)


class ContinualRandRegModel(RandRegModel):
    def get_embeddings(self, inputs, embedding_size):
        embeddings = [tf.Variable(
            tf.random_uniform([self.args.base_itput_num, embedding_size], -1.0,
                              1.0))]
        for i in range(self.V - self.args.base_itput_num):
            with tf.variable_scope("stage" + str(i + 1)):
                embeddings.append(tf.Variable(
                    tf.random_uniform([1, embedding_size], -1.0, 1.0)))
        embeddings = tf.concat(embeddings, 0)
        embed = tf.nn.embedding_lookup(embeddings, inputs)
        return embed, embeddings

    def get_representations(self):
        # embedding
        if self.args.use_embedding:
            with tf.variable_scope("primitive_encoding"):
                primitive, self.embeddings_primitive = self.get_embeddings(
                    self.x, self.args.embedding_size)
            with tf.variable_scope("functional_encoding"):
                function, self.embeddings_function = self.get_embeddings(
                    self.x, self.args.function_embedding_size)

        # compute switch
        x_one_hot = tf.one_hot(self.x, self.V, name='x_one_hot')
        with tf.variable_scope("compute_switch"):
            switch_score = self.ff(x_one_hot, 1, 32, 1)
            self.switch = tf.nn.sigmoid(
                switch_score / self.args.switch_temperature)

        return primitive, function

    def get_optimizer(self, scopes=None):
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

            if scopes is None:
                params = tf.trainable_variables(None)
            else:
                params = []
                for scope in scopes:
                    params += tf.trainable_variables(scope)
            gradients = tf.gradients(self.loss, params)
            self.params = params
            self.gradients = gradients
            clipped_gradients, _ = tf.clip_by_global_norm(
                gradients, self.args.max_gradient_norm)

            if self.args.decay_steps <= 0:
                optimizer_op = optimizer.apply_gradients(
                    zip(clipped_gradients, params))
            else:
                optimizer_op = optimizer.apply_gradients(
                    zip(clipped_gradients, params), global_step=global_step)
        return optimizer, optimizer_op

    def initialize(self, voc_size, act_size, stages):
        self.V = voc_size
        self.U = act_size
        self.regularization_list = []
        self.create_model()

        optimizer, self.optimizer_op = self.get_optimizer()
        self.continual_optimizer_op_list = []
        for i in range(stages):
            stage = i + 1
            if self.args.continual_all_params:
                self.continual_optimizer_op_list.append(self.optimizer_op)
            else:
                stage_id = "stage" + str(stage) + "/"
                scopes = []
                scopes.append('word_embeddings/primitive_encoding/' + stage_id)
                scopes.append(
                    'word_embeddings/functional_encoding/' + stage_id)
                scopes.append("prediction/" + stage_id)
                _, continual_optimizer_op = self.get_optimizer(scopes=scopes)
                self.continual_optimizer_op_list.append(continual_optimizer_op)

        init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

        tf.summary.FileWriter('logs/' + self.args.experiment_id + '/model',
                              self.sess.graph)
        self.sess.run(init)

    def create_model(self):
        sliced_y = self.create_placeholders()

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

            W = [tf.Variable(tf.random_uniform(
                [self.args.embedding_size, self.args.base_output_num], -1.0,
                1.0))]
            for i in range(self.U - self.args.base_output_num):
                with tf.variable_scope("stage" + str(i + 1)):
                    W.append(tf.Variable(
                        tf.random_uniform([self.args.embedding_size, 1], -1.0,
                                          1.0)))
            self.W = tf.concat(W, 1)
            h_matrix = tf.reshape(h, [-1, self.args.embedding_size])
            l_matrix = tf.matmul(h_matrix, self.W)
            l = tf.reshape(l_matrix, [-1, tf.shape(h)[1], self.U])
            l, _ = tf.split(l,
                            [self.valid_outputs, self.U - self.valid_outputs],
                            2)

        with tf.variable_scope("evaluation"):
            # loss
            loss_sum = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=sliced_y, logits=l)
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
            word_equality = tf.to_float(tf.equal(sliced_y, self.prediction))
            valid_word_equality = word_equality * self.target_mask_float
            self.word_accuracy = tf.reduce_mean(tf.reduce_sum(
                valid_word_equality, -1) / (tf.to_float(self.y_len)))

            # sentence accuracy
            sent_equality = tf.reduce_min(word_equality, axis=-1)
            self.sent_accuracy = tf.reduce_mean(sent_equality)

    def train(self, X, Y, X_len, Y_len, stage, valid_outputs,
              noise_weight=None, continual=False):
        log_step = 100

        if noise_weight is None:
            noise_weight = self.args.noise_weight

        if continual:
            optimizer_op = self.continual_optimizer_op_list[stage - 1]
            batch_size = self.args.continual_batch_size
            epochs = self.args.continual_epochs
        else:
            optimizer_op = self.optimizer_op
            batch_size = self.args.batch_size
            epochs = self.args.epochs
        fetch = [optimizer_op, self.loss, self.word_accuracy,
                 self.sent_accuracy]

        start = 0
        avg_loss, avg_wa, avg_sa = 0, 0, 0
        steps = 0
        for i in range(epochs):
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
                    self.noise_weight: noise_weight,
                    self.valid_outputs: valid_outputs}
            _, loss, word_acc, sent_acc = self.sess.run(fetch, feed_dict=feed)
            avg_loss += loss
            avg_wa += word_acc
            avg_sa += sent_acc
            steps += 1
            if i % log_step == 0 or i == epochs - 1:
                print(
                    i, avg_loss / steps, avg_wa / steps,
                       avg_sa / steps)
                avg_loss, avg_wa, avg_sa = 0, 0, 0
                steps = 0

    def test(self, X, Y, X_len, Y_len, valid_outputs, name, noise_weight=0.0):
        data_size = len(X)
        batch_size = self.args.test_batch_size
        loss, word_scc, sent_acc = 0.0, 0.0, 0.0
        prediction, attention, switch = [], [], []
        for start in range(0, data_size, batch_size):
            end = min(data_size, start + batch_size)
            feed = {self.x: X[start:end], self.y: Y[start:end],
                    self.x_len: X_len[start:end], self.y_len: Y_len[start:end],
                    self.noise_weight: noise_weight,
                    self.valid_outputs: valid_outputs}
            if start == 0:
                fetch = [self.loss, self.word_accuracy, self.sent_accuracy,
                         self.prediction, self.attention,
                         self.switch]
                b_loss, b_word_scc, b_sent_acc, b_prediction, b_attention, b_switch = self.sess.run(
                    fetch,
                    feed_dict=feed)
                prediction.extend(b_prediction)
                attention.extend(b_attention)
                switch.extend(b_switch)
            else:
                fetch = [self.loss, self.word_accuracy, self.sent_accuracy]
                b_loss, b_word_scc, b_sent_acc = self.sess.run(fetch,
                                                               feed_dict=feed)
            b_size = end - start
            loss += b_size * b_loss
            word_scc += b_size * b_word_scc
            sent_acc += b_size * b_sent_acc
        loss /= data_size
        word_scc /= data_size
        sent_acc /= data_size

        print(name, loss, word_scc, sent_acc)
        return prediction, attention, switch

    def get_embedding(self):
        fetch = [self.embeddings_primitive, self.embeddings_function, self.W]
        embeddings_primitive, embeddings_function, W = self.sess.run(fetch)
        return embeddings_primitive, embeddings_function, W

    def save_model(self, filename):
        save_path = self.saver.save(self.sess, filename)
        print("Model saved in path: %s" % save_path)


class NormalModel(RandRegModel):
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

        score = self.ff(logits, 1, 32, self.U)
        return score

    def create_model(self):
        sliced_y = self.create_placeholders()

        with tf.variable_scope("compute_switch"):
            switch_score = self.get_embeddings(self.x, 1)
            self.switch = tf.nn.sigmoid(
                switch_score / self.args.switch_temperature)
        # masks
        self.target_mask_float = tf.sequence_mask(
            self.y_len, maxlen=self.max_output_length, dtype=tf.float32)
        self.target_mask_int64 = tf.sequence_mask(
            self.y_len, maxlen=self.max_output_length, dtype=tf.int64)
        self.input_mask_float = tf.sequence_mask(
            self.x_len, maxlen=self.max_input_length, dtype=tf.float32)

        with tf.variable_scope("word_embeddings"):
            embedding = self.get_embeddings(self.x, 1)

        with tf.variable_scope("generate_attention"):
            self.attention = self.attention_generation(embedding)
        l = self.attention

        with tf.variable_scope("evaluation"):
            # loss
            loss_sum = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=sliced_y, logits=l)
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
            word_equality = tf.to_float(tf.equal(sliced_y, self.prediction))
            valid_word_equality = word_equality * self.target_mask_float
            self.word_accuracy = tf.reduce_mean(tf.reduce_sum(
                valid_word_equality, -1) / (tf.to_float(self.y_len)))

            # sentence accuracy
            sent_equality = tf.reduce_min(word_equality, axis=-1)
            self.sent_accuracy = tf.reduce_mean(sent_equality)


class ContinualNormalModel(ContinualRandRegModel):
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
        return logits

    def create_model(self):
        sliced_y = self.create_placeholders()

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
        h = self.attention

        with tf.variable_scope("prediction"):
            W = [tf.Variable(tf.random_uniform(
                [self.args.num_units, self.args.base_output_num], -1.0, 1.0))]
            for i in range(self.U - self.args.base_output_num):
                with tf.variable_scope("stage" + str(i + 1)):
                    W.append(tf.Variable(
                        tf.random_uniform([self.args.num_units, 1], -1.0,
                                          1.0)))
            self.W = tf.concat(W, 1)
            h_matrix = tf.reshape(h, [-1, self.args.num_units])
            l_matrix = tf.matmul(h_matrix, self.W)
            l = tf.reshape(l_matrix, [-1, tf.shape(h)[1], self.U])
            l, _ = tf.split(l,
                            [self.valid_outputs, self.U - self.valid_outputs],
                            2)
            self.l = l

        with tf.variable_scope("evaluation"):
            # loss
            loss_sum = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=sliced_y, logits=l)
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
            word_equality = tf.to_float(tf.equal(sliced_y, self.prediction))
            valid_word_equality = word_equality * self.target_mask_float
            self.word_accuracy = tf.reduce_mean(tf.reduce_sum(
                valid_word_equality, -1) / (tf.to_float(self.y_len)))

            # sentence accuracy
            sent_equality = tf.reduce_min(word_equality, axis=-1)
            self.sent_accuracy = tf.reduce_mean(sent_equality)


class EWCModel(ContinualNormalModel):
    def get_local_gradients(self):
        return self.gradients

    def initialize(self, voc_size, act_size, stages):
        ret = super(EWCModel, self).initialize(voc_size, act_size, stages)
        gradients = self.get_local_gradients()
        self.static_gradients = []
        for g in gradients:
            if g is None:
                self.static_gradients.append(g)
            else:
                self.static_gradients.append(tf.square(tf.stop_gradient(g)))
        return ret

    def train(self, X, Y, X_len, Y_len, stage, valid_outputs,
              noise_weight=None, continual=False):
        super(EWCModel, self).train(X, Y, X_len, Y_len, stage, valid_outputs,
                                    noise_weight, continual)

        # extend loss
        assert len(self.params) == len(self.static_gradients)
        current_params = []
        current_gradients = []
        for p, g in zip(self.params, self.static_gradients):
            if p is None or g is None:
                continue
            if '/stage' in p.name:
                if 'function' in p.name:
                    continue
                terms = p.name.split('/')
                for term in terms:
                    if len(term) > 5 and 'stage' == term[:5]:
                        s = int(term[5:])
                        if s <= stage:
                            current_params.append(p)
                            current_gradients.append(g)
                        break
            else:
                current_params.append(p)
                current_gradients.append(g)

        param_v = self.sess.run(current_params)

        squares = []
        for a, g, b in zip(current_params, current_gradients, param_v):
            diff = tf.square(a - b)
            reg = g * diff
            squares.append(tf.reduce_sum(reg))
        self.loss += self.args.baseline_lambda * tf.reduce_sum(squares)

    def test(self, X, Y, X_len, Y_len, valid_outputs, name, noise_weight=0.0):
        loss = self.loss
        self.loss = self.word_accuracy
        ret = super(EWCModel, self).test(X, Y, X_len, Y_len, valid_outputs,
                                         name, noise_weight)
        self.loss = loss
        return ret


class MASModel(EWCModel):
    def get_local_gradients(self):
        f_loss = tf.reduce_sum(tf.square(self.l))
        gradients = tf.gradients(f_loss, self.params)
        return gradients
