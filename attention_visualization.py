import argparse
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rc


def filter_words(words):
    result = []
    for word in words:
        if word == "EOS_X":
            result.append("<EOS>")
        else:
            result.append(word)
    return result


def filter_actions(actions):
    result = []
    for action in actions:
        if action == "EOS_Y":
            result.append("<EOS>")
        elif action == "switch":
            result.append("(switch)")
        else:
            if action[:2] == "I_":
                action = action[2:]
            if action == "TURN_RIGHT":
                action = "RTURN"
            elif action == "TURN_LEFT":
                action = "LTURN"
            result.append(action)
    return result


def visualization(args, text_file, image_file):
    matrix = []
    with open(text_file) as f:
        lines = f.readlines()
        a = lines[0].strip().split(' ')[1:]
        c = []
        for line in lines[1:]:
            terms = line.strip().split(' ')
            c.append(terms[0])
            matrix.append([float(x) for x in terms[1:len(a) + 1]])

    if args.hide_switch:
        matrix = matrix[1:]
        c = c[1:]

    a = filter_words(a)
    c = filter_actions(c)

    if len(matrix) == 0:
        return

    fig = plt.figure(figsize=[9.6, 9.6])
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, cmap='bone')
    if not args.hide_bar:
        fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + a, rotation=90)
    ax.set_yticklabels([''] + c)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.tight_layout()

    plt.savefig(image_file)
    plt.clf()
    plt.close('all')


def main(args):
    directory1 = "logs/" + args.experiment_id + "/" + args.input_folder
    directory2 = "logs/" + args.experiment_id + "/" + args.output_folder
    num_samples = len(os.listdir(directory1))
    num_samples = min(args.num_samples, num_samples)

    if not os.path.exists(directory2):
        os.makedirs(directory2)

    font = {'family': 'serif',
            'size': 36}

    rc('font', **font)

    for i in range(num_samples):
        text_file = directory1 + "/" + str(i) + ".txt"
        image_file = directory2 + "/" + str(i) + ".pdf"
        visualization(args, text_file, image_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attention Visualization.')
    parser.add_argument('--experiment_id', type=str, default='sanity',
                        help='experiment ID')
    parser.add_argument('--input_folder', type=str, default='attention',
                        help='name of input folder')
    parser.add_argument('--output_folder', type=str, default='attention_vis',
                        help='name of output folder')
    parser.add_argument('--hide_switch', action='store_true', default=False,
                        help='visualize_switch.')
    parser.add_argument('--hide_bar', action='store_true', default=False,
                        help='hide scale bar.')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='number of samples to visualize')
    args = parser.parse_args()

    main(args)
