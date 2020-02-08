import argparse
import os
import random


def combine(args, temp, pin, pout):
    temp = temp.replace(args.input_key, pin).replace(args.output_key, pout)
    return temp.strip()


def get_eval_template(args):
    with open(os.path.join(args.input_dir, 'test.txt'), 'r') as f:
        lines = f.readlines()
    ret = []
    samples = []
    for line in lines:
        line = line.strip()
        samples.append(line)
    return ret, samples


def write_file(args, filename, data):
    random.shuffle(data)
    if args.eval_size_limit >= 0 and len(data) > args.eval_size_limit:
        data = data[:args.eval_size_limit]
    with open(filename, 'w') as f:
        for entry in data:
            f.write(entry.strip() + '\n')


def generate(args, train_template, eval_template, forget_samples, i):
    output_dir = os.path.join(args.output_dir, str(i))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    en = args.input_key + "_" + str(i)
    fr = args.output_key + "_" + str(i)

    # train
    train_sample = combine(args, train_template, en, fr)
    write_file(args, os.path.join(output_dir, 'train.txt'), [train_sample])

    # eval1
    eval1_samples = []
    for template in eval_template:
        eval1_samples.append(combine(args, template, en, fr))
    write_file(args, os.path.join(output_dir, 'eval1.txt'), eval1_samples)

    if i == 1:
        write_file(args, os.path.join(output_dir, 'eval2.txt'), eval1_samples)
    else:
        write_file(args, os.path.join(output_dir, 'eval2.txt'), forget_samples)
    forget_samples.extend(eval1_samples)


def get_initial_data(args):
    output_dir = os.path.join(args.output_dir, '0')
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # train
    with open(os.path.join(args.input_dir, 'train.txt'), 'r') as f:
        lines = f.readlines()
    key = " " + args.input_key + " "
    train_template = None
    train = [line.strip() for line in lines]
    for line in lines:
        if key in line and train_template is None:
            train_template = line.strip()
    assert train_template is not None

    write_file(args, os.path.join(output_dir, 'train.txt'), train)

    with open(os.path.join(args.input_dir, 'test.txt'), 'r') as f:
        lines = f.readlines()
    test = [line.strip() for line in lines]
    write_file(args, os.path.join(output_dir, 'eval1.txt'), test)
    write_file(args, os.path.join(output_dir, 'eval2.txt'), test)
    return test, train_template


def get_params(args):
    args.input_dir = os.path.join('data_orig', args.data_name)
    args.output_dir = os.path.join('data', args.data_name)
    if args.data_name == 'data_scan':
        args.input_key = 'jump'
        args.output_key = 'JUMP'
    elif args.data_name == 'data_translate':
        args.input_key = 'daxy'
        args.output_key = 'daxiste'
    elif args.data_name == 'data_adj':
        args.input_key = 'rubber'
        args.output_key = 'RUBBER'
    elif args.data_name == 'data_fewshot':
        args.input_key = 'zup'
        args.output_key = 'YYY'
    else:
        assert False
    return args


def main(args):
    random.seed(12)
    args = get_params(args)
    _, eval_template = get_eval_template(args)

    # initial
    forget_samples, train_template = get_initial_data(args)

    for i in range(1, 101):
        generate(args, train_template, eval_template, forget_samples, i)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Generator.')
    parser.add_argument('--full_data', action='store_true', default=False,
                        help='Use all data.')
    parser.add_argument('--num_stages', type=int, default=100,
                        help='number of stages.')
    parser.add_argument('--eval_size_limit', type=int, default=100000,
                        help='maximum limit of evaluation data size.')
    parser.add_argument('--data_name', type=str,
                        default='data_scan', help='input dir')
    args = parser.parse_args()
    main(args)
