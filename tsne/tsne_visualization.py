import os
import argparse
import numpy as np
import pylab
from matplotlib import rc

from tsne import tsne


def load_dict(fn):
    with open(fn, 'r') as f:
        lines = f.readlines()
    ret = ['<EOS>']
    for line in lines:
        ret.append(line.strip().split()[1])
    return ret


def draw_figures_all(args, Y, labels, basedir):
    initial_stage_size = args.initial_stage_size
    max_elements = args.max_elements
    colors = []
    for i in range(len(Y)):
        stage = max(0, i - initial_stage_size)
        color = (stage + 9) / 10
        colors.append(color)

    annotations = []
    for i in range(len(Y)):
        anno = str(max(0, i - initial_stage_size))
        if i < max_elements:
            anno += ":" + labels[i]
        annotations.append(anno)

    font_size = 16
    pylab.tick_params(axis='both', which='major', labelsize=font_size)
    pylab.scatter(Y[:, 0], Y[:, 1], 20, np.asarray(colors))

    for i, txt in enumerate(annotations):
        pylab.annotate(txt, (Y[:, 0][i], Y[:, 1][i]), fontsize=12)

    pylab.savefig(basedir + '.pdf')


def draw_one_batch(Y, colors, annotations, basedir, labels, spl=None, x_lim=None, y_lim=None, ):
    font_size = 18
    pylab.tick_params(axis='both', which='major', labelsize=font_size)
    if spl is None:
        pylab.scatter(Y[:, 0], Y[:, 1], 20, np.asarray(colors))
    else:
        pylab.scatter(Y[:, 0][:spl], Y[:, 1][:spl], 30, np.asarray(colors[:spl]), marker='o', label=labels[0])
        pylab.scatter(Y[:, 0][spl:], Y[:, 1][spl:], 60, facecolors='none', edgecolors=np.asarray(colors[spl:]), marker='^', label=labels[1])
    axes = pylab.gca()
    if x_lim is None and y_lim is None:
        x_lim = axes.get_xlim()
        y_lim = axes.get_ylim()
    else:
        axes.set_xlim(x_lim)
        axes.set_ylim(y_lim)

    for i, txt in enumerate(annotations):
        if len(txt) > 0:
            pylab.annotate(txt, (Y[:, 0][i], Y[:, 1][i]), fontsize=12)

    axes.legend(loc='lower right', framealpha=0.5, fontsize=font_size)
    pylab.savefig(basedir + '.pdf', bbox_inches='tight', pad_inches=0.1)

    pylab.clf()

    return x_lim, y_lim


def draw_figures_separate(args, Y, labels, basedir):
    initial_stage_size = args.initial_stage_size
    max_elements = args.max_elements
    colors = []
    for i in range(len(Y)):
        stage = max(0, i - initial_stage_size)
        color = (stage + 9) / 10
        colors.append(color)

    annotations = []
    for i in range(len(Y)):
        anno = str(max(0, i - initial_stage_size))
        if anno == '0':
            anno = ''
        if i < max_elements:
            if anno == '':
                anno = labels[i]
            else:
                anno += ":" + labels[i]
        annotations.append(anno)

    num_figures = 4
    num_stages = len(Y) - initial_stage_size
    batch_size = int(num_stages / num_figures)
    if num_stages % num_figures > 0:
        batch_size += 1

    x_lim, y_lim = None, None
    for i in range(num_figures - 1, -1, -1):
        a_size = initial_stage_size + i * batch_size
        b_size = min(len(Y) - a_size, batch_size)
        all_size = a_size + b_size
        Y_local = Y[:all_size]
        if i == 0:
            annotations_local = annotations[:a_size]
        else:
            annotations_local = [''] * a_size
        annotations_local.extend([''] * b_size)
        local_colors = ['b'] * a_size + ['r'] * b_size
        basename = os.path.basename(basedir)
        local_dir = basedir + '/' + basename + '_' + str(i)
        if i == 0:
            labels = ['0']
        else:
            labels = ['0-' + str(a_size - initial_stage_size)]
        labels.append(str(a_size - initial_stage_size + 1) + '-' + str(all_size - initial_stage_size))
        x_lim, y_lim = draw_one_batch(Y_local, local_colors, annotations_local, local_dir, labels, a_size, x_lim, y_lim)


def main(args):
    font = {'family': 'serif'}
    rc('font', **font)

    basedir = args.output_prefix
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    output_txt_file = basedir + '.txt'

    if args.load_tsne_output and os.path.exists(output_txt_file):
        Y = []
        labels = []
        with open(output_txt_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            terms = line.strip().split('\t')
            labels.append(terms[0])
            Y.append([float(terms[1]), float(terms[2])])
        Y = np.asarray(Y)
    else:
        X = np.loadtxt(args.input_file)
        if args.transpose:
            X = X.transpose()
        Y = tsne(X, 2, 50, 20.0)
        labels = load_dict(args.dict_file)
        with open(output_txt_file, 'w') as f:
            for a, y in zip(labels, Y):
                f.write(a + '\t' + str(y[0]) + '\t' + str(y[1]) + '\n')

    if args.separate_plot:
        draw_figures_separate(args, Y, labels, basedir)
    else:
        draw_figures_all(args, Y, labels, basedir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Continual Learning Attention Visualization.')
    parser.add_argument('--input_file', type=str, default='primitive.txt',
                        help='input file.')
    parser.add_argument('--dict_file', type=str, default='voc_dict.txt',
                        help='dictionary file.')
    parser.add_argument('--output_prefix', type=str, default='./primitive_visualization',
                        help='dictionary file.')
    parser.add_argument('--transpose', action='store_true', default=False,
                        help='transpose input.')
    parser.add_argument('--load_tsne_output', action='store_true', default=False,
                        help='load t-SNE output.')
    parser.add_argument('--separate_plot', action='store_true', default=False,
                        help='plot data to separate files.')
    parser.add_argument('--initial_stage_size', type=int, default=13,
                        help='number of elements in initial stage')
    parser.add_argument('--max_elements', type=int, default=6,
                        help='maximum number of elements to show name')
    args = parser.parse_args()

    main(args)
