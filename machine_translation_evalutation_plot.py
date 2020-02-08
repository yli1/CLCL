import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


def draw(lists, stds, legends, basedir, colors, lw, loc, plot=True):
    directory = os.path.dirname(basedir)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(basedir + '.txt', 'w') as f:
        for i in range(len(lists[0])):
            f.write(str(i + 1))
            for e in lists:
                f.write('\t' + str(e[i]))
            for e in stds:
                f.write('\t' + str(e[i]))
            f.write('\n')

    if not plot:
        return

    plt.figure(figsize=(9, 6))
    ax = plt.subplot(1, 1, 1)
    font_size = 24
    ax.tick_params(axis='both', which='major', labelsize=font_size)

    ax.axhline(100, lw=lw, c='lightgray', ls='--', zorder=0)
    ax.axhline(0, lw=lw, c='lightgray', ls='--', zorder=0)
    for i, (l, legend) in enumerate(zip(lists, legends)):
        color_index = min(i, len(colors) - 1)
        entries = colors[color_index]
        color, marker = entries
        l1 = [0]
        l1.extend(l)
        ax.plot(l1, lw=lw, markevery=(10, 20), marker=marker, markersize=16,
                markeredgewidth=2, markerfacecolor='none', color=color,
                label=legend)

    ax.set_xlim([1, 100])
    ax.legend(loc=loc, prop={'size': font_size})
    ax.set_xlabel('Stage', fontsize=font_size)
    ax.set_ylabel('Accuracy (%)', fontsize=font_size)
    ax.xaxis.labelpad = -2
    ax.yaxis.labelpad = -12
    plt.savefig(basedir + '.pdf', bbox_inches='tight', pad_inches=0.01)


def get_list(lines, key):
    ret = []
    for line in lines:
        if key not in line:
            continue
        str_value = line.strip().split(',')[3][1:-1]
        ret.append(100 * float(str_value))
    return ret


def load(fn):
    if os.path.exists(fn):
        with open(fn, 'r') as f:
            lines = f.readlines()
        eval1 = get_list(lines, 'Eval1 in stage')
        eval2 = get_list(lines, 'Eval2 in stage')
        eval3 = get_list(lines, 'Eval3 in stage')
        eval4 = get_list(lines, 'Initial forgetting')
    else:
        eval1 = []
        eval2 = []
        eval3 = []
        eval4 = []
    return eval1, eval2, eval3, eval4


def get_results(args, path):
    if args.first_experiment:
        exps = ['a']
    else:
        exps = ['a', 'b', 'c', 'd', 'e']

    results = [[], [], [], []]
    for e in exps:
        fn = os.path.join(path + e, "log.txt")
        eval1, eval2, eval3, eval4 = load(fn)
        results[0].append(eval1)
        results[1].append(eval2)
        results[2].append(eval3)
        results[3].append(eval4)

    for r in results[0]:
        assert len(results[0][0]) == len(r)

    means = []
    stds = []
    for result in results:
        matrix = np.asarray(result)
        means.append(np.mean(matrix, axis=0))
        stds.append(np.std(matrix, axis=0))

    return means, stds


def get_params(args, analysis=False):
    if not analysis:
        pairs = [
            ('Standard', 'logs/baseline_Standard_translate.', ('orange', 'v')),
            ('Compositional', 'logs/baseline_Compositoinal_translate.',
             ('r', '^')),
            ('EWC', 'logs/baseline_EWC_translate.', ('b', 'D')),
            ('MAS', 'logs_mas/baseline_MAS_translate.', ('c', 's')),
            ("$\mathbf{Proposed}$", 'logs/main_proposed_translate.',
             ('g', 'o')),
        ]
        file_list = [x[1] for x in pairs]
        legends = [x[0] for x in pairs]
        colors = [x[2] for x in pairs]
        output_list = [
            'outputs_translation/translate_compositional_generalization_results',
            'outputs_translation/translate_catastrophic_forgetting_results',
            'outputs_translation/translate_long_term_catastrophic_forgetting_results',
            'outputs_translation/translate_initial_results']
        lw = 2
        loc = 'center right'
    else:
        pairs = [
            ('$k_p$=4', 'logs/embedding_4.', ('orange', 'v')),
            ('$k_p$=8', 'logs/embedding_8.', ('r', '^')),
            ('$k_p$=16', 'logs/embedding_16.', ('m', 'x')),
            ('$k_p$=32', 'logs/embedding_32.', ('b', 'D')),
            ('$k^*_p$=64', 'logs/main_proposed.', ('g', 'o')),
            ('$k_p$=128', 'logs/embedding_128.', ('c', 's')),
        ]
        file_list = [x[1] for x in pairs]
        legends = [x[0] for x in pairs]
        colors = [x[2] for x in pairs]
        output_list = [
            'outputs_translation/translate_compositional_generalization_analysis',
            'outputs_translation/translate_catastrophic_forgetting_analysis',
            'outputs_translation/translate_long_term_catastrophic_forgetting_analysis',
            'outputs_translation/translate_initial_analysis']
        lw = 2
        loc = 'center right'
    return file_list, legends, output_list, colors, lw, loc


def main(args):
    file_list, legends, output_list, colors, lw, loc = get_params(args,
                                                                  args.analysis)
    eval1_list = []
    eval2_list = []
    eval3_list = []
    eval4_list = []
    std1_list = []
    std2_list = []
    std3_list = []
    std4_list = []
    for fn in file_list:
        means, stds = get_results(args, fn)
        eval1, eval2, eval3, eval4 = means
        std1, std2, std3, std4 = stds
        eval1_list.append(eval1)
        eval2_list.append(eval2)
        eval3_list.append(eval3)
        eval4_list.append(eval4)
        std1_list.append(std1)
        std2_list.append(std2)
        std3_list.append(std3)
        std4_list.append(std4)

    font = {'family': 'serif'}
    rc('font', **font)

    draw(eval1_list, std1_list, legends, output_list[0], colors, lw, loc)
    draw(eval2_list, std2_list, legends, output_list[1], colors, lw, loc)
    draw(eval3_list, std3_list, legends, output_list[2], colors, lw, loc)
    draw(eval4_list, std4_list, legends, output_list[3], colors, lw, loc,
         plot=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Continual Learning evaluation.')
    parser.add_argument('--analysis', action='store_true', default=False,
                        help='Analysis.')
    parser.add_argument('--first_experiment', action='store_true',
                        default=False,
                        help='Visualize first experiment.')
    args = parser.parse_args()
    main(args)
