import argparse
import random
import editdistance
import os
import numpy as np

from data_generator import ContinualLearningFormarter
from data_generator import get_data_generator
from model import get_model


def visualization(i, a, b, c, d, e, directory1):
    with open(directory1 + '/' + str(i) + '.txt', 'w') as f:
        a = a + ["EOS_X"]
        c = c + ["EOS_Y"]
        f.write(str(i) + ' ')
        f.write(' '.join(a))
        f.write('\n')
        f.write('switch ')
        f.write(' '.join(str(x[0]) for x in e))
        f.write('\n')
        for p, q in zip(c, d):
            f.write(p + ' ')
            f.write(' '.join(str(x) for x in q))
            f.write('\n')


def evaluation(test_X, test_Y, prediction, attention, switch, act, fn,
               suffix=''):
    id2act = {i: a for a, i in act.items()}
    actions = []
    for pred in prediction:
        acts = []
        for id in pred:
            if id == 0:
                break
            acts.append(id2act[id])
        actions.append(acts)

    directory = fn + "/attention" + suffix
    if not os.path.exists(directory):
        os.makedirs(directory)

    avg_wer = 0
    with open(fn + '/output' + suffix + '.txt', 'w') as f:
        for i, (a, b, c, d, e) in enumerate(
                zip(test_X, test_Y, actions, attention, switch)):
            ed = editdistance.eval(b, c)
            wer = ed / float(len(b))
            avg_wer += wer

            f.write(str(i) + '\t')
            f.write(str(len(b)) + '\t')
            f.write(str(len(c)) + '\t')
            f.write(str(ed) + '\t')
            f.write(str(wer))
            f.write('\n')
            f.write(' '.join(a))
            f.write('\n')
            f.write(' '.join(str(x[0]) for x in e))
            f.write('\n')
            f.write(' '.join(b))
            f.write('\n')
            f.write(' '.join(c))
            f.write('\n\n')
            visualization(i, a, b, c, d, e, directory)
    print('Average Word Error Rate: ' + str(avg_wer / len(test_Y)))


def visualize_embedding(filename, embedding):
    with open(filename, 'w') as f:
        for row in embedding:
            for element in row:
                f.write(str(element) + '\t')
            f.write('\n')


def visualize_parameters(model, directory, model_directory, stage):
    embeddings_primitive, embeddings_function, W = model.get_embedding()
    directory = directory + '/' + str(stage)
    if not os.path.exists(directory):
        os.makedirs(directory)
    visualize_embedding(directory + '/primitive.txt', embeddings_primitive)
    visualize_embedding(directory + '/function.txt', embeddings_function)
    visualize_embedding(directory + '/prediction.txt', W)

    if stage < 4 or stage % 10 == 0 or stage == args.stages:
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        model.save_model(model_directory + '/' + str(stage) + '/model.ckpt')


def write_dict(filename, dict):
    print(dict)
    with open(filename, 'w') as f:
        ret = []
        for key, value in dict.items():
            ret.append((value, key))
        ret.sort()
        for (value, key) in ret:
            f.write(str(value) + '\t' + str(key) + '\n')


def get_converted_data(type, stage, dg, formater, dicts, maxs, sort=False):
    if type == "train":
        X, Y, X_len, Y_len = dg.get_train_data_convert(stage, formater, dicts,
                                                       maxs)
    elif type == "eval1":
        X, Y, X_len, Y_len = dg.get_eval1_data_convert(stage, formater, dicts,
                                                       maxs)
    elif type == "eval2":
        X, Y, X_len, Y_len = dg.get_eval2_data_convert(stage, formater, dicts,
                                                       maxs)
    else:
        raise ValueError("Type is not defined: " + type)

    if sort:
        slist = zip(Y_len, X, Y, X_len)
        slist.sort()
        Y_len = [e[0] for e in slist]
        X = [e[1] for e in slist]
        Y = [e[2] for e in slist]
        X_len = [e[3] for e in slist]

    return X, Y, X_len, Y_len


def process_continual(args):
    # get data generator
    dg = get_data_generator(args.data_name, args)
    log_dir = 'logs/' + args.experiment_id
    param_dir = log_dir + '/params'
    model_dir = log_dir + '/checkpoints'

    # get dictionary
    X, Y = [], []
    for i in range(args.stages + 1):
        X_i, Y_i = dg.get_train_data(i)
        if i == 0:
            init_X, init_Y = X_i, Y_i
        X.extend(X_i)
        Y.extend(Y_i)

    formater = ContinualLearningFormarter(args)
    dicts, maxs = formater.initialize(X, Y)
    voc, act = dicts
    write_dict(log_dir + '/voc_dict.txt', voc)
    write_dict(log_dir + '/act_dict.txt', act)
    max_input, max_output = maxs

    formater_init = ContinualLearningFormarter(args)
    dicts_init, _ = formater_init.initialize(init_X, init_Y)
    voc_init, act_init = dicts_init

    # initialize model
    base_itput_num = len(voc_init) + 1  # 13
    base_output_num = len(act_init) + 1  # 6
    args.base_itput_num = base_itput_num
    args.base_output_num = base_output_num
    args.input_length = max_input
    args.output_length = max_output
    model = get_model(args.model_name, args)
    model.initialize(len(voc) + 1, len(act) + 1, args.stages)

    noise_weight = args.noise_weight

    # initial stage
    visualize_parameters(model, param_dir, model_dir, -1)
    X, Y, X_len, Y_len = get_converted_data("train", 0, dg, formater, dicts,
                                            maxs)
    model.train(X, Y, X_len, Y_len, 0, base_output_num,
                noise_weight=noise_weight)
    visualize_parameters(model, param_dir, model_dir, 0)

    model.test(X, Y, X_len, Y_len, base_output_num, "Initial result")

    X, Y, X_len, Y_len = get_converted_data("eval1", 0, dg, formater, dicts,
                                            maxs, sort=True)
    model.test(X, Y, X_len, Y_len, base_output_num, "Initial compositionality")

    X, Y, X_len, Y_len = get_converted_data("eval2", 0, dg, formater, dicts,
                                            maxs, sort=True)
    model.test(X, Y, X_len, Y_len, base_output_num, "Initial forgetting")

    # continual stages
    for i in range(args.stages):
        stage = i + 1
        valid_outputs = base_output_num + stage
        evaluation_stage = (
                    stage == 1 or stage == 2 or stage == 10 or stage == args.stages)

        noise_weight *= args.noise_weight_decay

        # train
        X, Y, X_len, Y_len = get_converted_data("train", stage, dg, formater,
                                                dicts, maxs)
        model.train(X, Y, X_len, Y_len, stage, valid_outputs,
                    noise_weight=noise_weight, continual=True)
        visualize_parameters(model, param_dir, model_dir, stage)

        # eval1
        X, Y, X_len, Y_len = get_converted_data("eval1", stage, dg, formater,
                                                dicts, maxs,
                                                sort=not evaluation_stage)
        prediction, attention, switch = model.test(X, Y, X_len, Y_len,
                                                   valid_outputs,
                                                   "Eval1 in stage " + str(
                                                       stage))
        if evaluation_stage:
            ori_test_X, ori_test_Y = dg.get_eval1_data(stage)
            evaluation(ori_test_X, ori_test_Y, prediction, attention, switch,
                       act, 'logs/' + args.experiment_id,
                       suffix='_eval1_' + str(stage))

        # eval2
        X, Y, X_len, Y_len = get_converted_data("eval2", stage, dg, formater,
                                                dicts, maxs,
                                                sort=not evaluation_stage)
        prediction, attention, switch = model.test(X, Y, X_len, Y_len,
                                                   valid_outputs,
                                                   "Eval2 in stage " + str(
                                                       stage))
        if evaluation_stage:
            ori_test_X, ori_test_Y = dg.get_eval2_data(stage)
            evaluation(ori_test_X, ori_test_Y, prediction, attention, switch,
                       act, 'logs/' + args.experiment_id,
                       suffix='_eval2_' + str(stage))

        # eval3
        X, Y, X_len, Y_len = get_converted_data("eval2", 0, dg, formater,
                                                dicts, maxs,
                                                sort=not evaluation_stage)
        # X, Y, X_len, Y_len = get_converted_data("eval2", 1, dg, formater, dicts, maxs, sort=not evaluation_stage)
        prediction, attention, switch = model.test(X, Y, X_len, Y_len,
                                                   valid_outputs,
                                                   "Eval3 in stage " + str(
                                                       stage))
        if evaluation_stage:
            ori_test_X, ori_test_Y = dg.get_eval2_data(0)
            # ori_test_X, ori_test_Y = dg.get_eval2_data(1)
            evaluation(ori_test_X, ori_test_Y, prediction, attention, switch,
                       act, 'logs/' + args.experiment_id,
                       suffix='_eval3_' + str(stage))


def main(args):
    seed = args.random_seed
    random.seed(seed)
    if args.random_random:
        np.random.seed(random.randint(2, 1000))
    else:
        np.random.seed(seed)

    # organizing parameters
    if args.remove_noise:
        args.noise_weight = 0.0

    process_continual(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compositional Instructions.')
    parser.add_argument('--experiment_id', type=str, default='default',
                        help='experiment ID')
    parser.add_argument('--model_name', type=str, default='continual_rand_reg',
                        help='model name')
    parser.add_argument('--print_output', action='store_true', default=False,
                        help='Linear max.')
    parser.add_argument('--simple_data', action='store_true', default=False,
                        help='use simple data.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--learning_rate', type=float, default=0.3,
                        help='learning rate')
    parser.add_argument('--batch_size', type=int, default=10,
                        help='batch_size')
    parser.add_argument('--continual_batch_size', type=int, default=1,
                        help='continual_batch_size')
    parser.add_argument('--test_batch_size', type=int, default=100000,
                        help='batch_size')
    parser.add_argument('--shuffle_batch', action='store_true', default=False,
                        help='shuffle batch.')
    parser.add_argument('--random_batch', action='store_true', default=False,
                        help='random batch.')
    parser.add_argument('--epochs', type=int, default=1,
                        help='epochs')
    parser.add_argument('--continual_epochs', type=int, default=100,
                        help='epochs')
    parser.add_argument('--data_name', type=str, default='continual',
                        help='name of data set')
    parser.add_argument('--train_file', type=str,
                        default='SCAN/add_prim_split/tasks_train_addprim_jump.txt',
                        help='train file name')
    parser.add_argument('--test_file', type=str,
                        default='SCAN/add_prim_split/tasks_test_addprim_jump.txt',
                        help='test file name')
    parser.add_argument('--switch_temperature', type=float, default=1.0,
                        help='switch temperature')
    parser.add_argument('--attention_temperature', type=float, default=10.0,
                        help='attention temperature')
    parser.add_argument('--num_units', type=int, default=16,
                        help='num units')
    parser.add_argument('--bidirectional_encoder', action='store_true',
                        default=False,
                        help='bidirectional encoder.')
    parser.add_argument('--max_gradient_norm', type=float, default=1.0,
                        help='max gradient norm')
    parser.add_argument('--decay_steps', type=int, default=-1,
                        help='decay steps')
    parser.add_argument('--use_input_length', action='store_true',
                        default=False,
                        help='use input length.')
    parser.add_argument('--use_embedding', action='store_true', default=True,
                        help='use embedding.')
    parser.add_argument('--embedding_size', type=int, default=8,
                        help='embedding size')
    parser.add_argument('--function_embedding_size', type=int, default=8,
                        help='embedding size')
    parser.add_argument('--reg_coe', type=float, default=-1.0,
                        help='regularization coeficient')
    parser.add_argument('--baseline_lambda', type=float, default=10.0,
                        help='regularization coeficient for baseline methods.')
    parser.add_argument('--macro_switch_reg_coe', type=float, default=-1.0,
                        help='macro switch regularization coeficient')
    parser.add_argument('--relu_switch', action='store_true', default=False,
                        help='relu switch')
    parser.add_argument('--use_start_symbol', action='store_true',
                        default=False,
                        help='use start symbol')
    parser.add_argument('--content_noise', action='store_true', default=False,
                        help='add noise to content')
    parser.add_argument('--content_noise_coe', type=float, default=-1.0,
                        help='noise regularization coeficient')
    parser.add_argument('--sample_wise_content_noise', action='store_true',
                        default=False,
                        help='sample-wise noise regularization')
    parser.add_argument('--noise_weight', type=float, default=1.0,
                        help='noise weight')
    parser.add_argument('--noise_weight_decay', type=float, default=1.0,
                        help='noise weight decay')
    parser.add_argument('--remove_noise', action='store_true', default=False,
                        help='remove noise')
    parser.add_argument('--function_noise', action='store_true', default=False,
                        help='add noise to function')
    parser.add_argument('--remove_x_eos', action='store_true', default=False,
                        help='remove x eos')
    parser.add_argument('--masked_attention', action='store_true',
                        default=False,
                        help='masked attention')
    parser.add_argument('--remove_switch', action='store_true', default=False,
                        help='remove switch')
    parser.add_argument('--use_entropy_reg', action='store_true',
                        default=False,
                        help='use entropy reg')
    parser.add_argument('--random_random', action='store_true', default=False,
                        help='random_random')
    parser.add_argument('--single_representation', action='store_true',
                        default=False,
                        help='single representation')
    parser.add_argument('--use_decoder_input', action='store_true',
                        default=False,
                        help='single representation')
    parser.add_argument('--output_embedding_size', type=int, default=8,
                        help='output embedding size')
    parser.add_argument('--use_l1_norm', action='store_true', default=False,
                        help='single representation')
    parser.add_argument('--continual_learning', action='store_true',
                        default=True,
                        help='continual learning')
    parser.add_argument('--train_file_2', type=str,
                        default='data_extended/continual/jump_only.txt',
                        help='second stage training data set')
    parser.add_argument('--test_file_2', type=str,
                        default='data_extended/continual/jump_included.txt',
                        help='second test data')
    parser.add_argument('--continual_all_params', action='store_true',
                        default=False,
                        help='continual learning with all parameters updated.')
    parser.add_argument('--remove_prediction_bias', action='store_true',
                        default=False,
                        help='remove prediction bias')
    parser.add_argument('--use_stage_data', action='store_true', default=False,
                        help='use stage data')
    parser.add_argument('--stages', type=int, default=10,
                        help='number of stages')
    parser.add_argument('--data_dir', type=str, default='data/data_scan',
                        help='data dir')
    args = parser.parse_args()

    main(args)
