# Copied from https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/visualize_atari.py
# and https://github.com/emansim/baselines-mansimov/blob/master/baselines/a2c/load.py
# Thanks to the author and OpenAI team!

import glob
import os

import matplotlib
# these two lines cause the micropolis gui to crash for some reason
import numpy as np
from scipy.signal import medfilt
matplotlib.rcParams.update({'font.size': 8})

# the index of particular columns in the log files
# TODO: we could make this depend directly on the ordering of columns defined
# in envs.py
header_idxs ={
        'r': 0,
        'e': 3,
        'p': 4
        }
header_names = {
        'r': 'Rewards',
        'e': 'Action Entropy',
        'p': 'Target Population'
        }

from imutils import paths
from graphviz import Digraph, Graph
color_defaults = [
    '#1f77b4',  # muted blue
    '#ff7f0e',  # safety orange
    '#2ca02c',  # cooked asparagus green
    '#d62728',  # brick red
    '#9467bd',  # muted purple
    '#8c564b',  # chestnut brown
    '#e377c2',  # raspberry yogurt pink
    '#7f7f7f',  # middle gray
    '#bcbd22',  # curry yellow-green
    '#17becf'   # blue-teal
]

def network_graphs():
    dot = Digraph(comment='StrictlyConv', node_attr={'shape': 'box'})
    dot.edge_attr.update(arrowhead='vee')
    dot.edge_attr.update(color=color_defaults[0])
    dot.node_attr.update(width='2', height='0.2')
    dot.node('A', 'Map', shape='box')
    dot.node('B', '')
    dot.node('C', '')
    dot.node('D', 'Action Map')
    dot.edge('A', 'B', label='c_{k5}')
    dot.edge('B', 'C', label='c_{k3}')
    dot.edge('C', 'D')
    dot.node('E', '', width='1')
    dot.edge('C', 'E', label='d')
    dot.node('F', '', width='0.5')
    dot.edge('E', 'F', label='d')
    dot.node('G', '', width='0.25')
    dot.edge('F', 'G', label='d')
    dot.node('H', '', xlabel='Scalar Value Prediction', width='0.125', shape='circle')
    dot.edge('G', 'H', label='d')
    dot.render('strictlyConv.gv', view=True)

    dott = Digraph(comment='FractalNet', node_attr={'shape': 'box'})
    def expand(dot, i):
        dott.edge_attr.update(arrowhead='vee')
        dott.edge_attr.update(color=color_defaults[0])
        dott.node_attr.update(width='2', height='0.2')
        with dot.subgraph(name='fc_{}'.format(i)) as subg:
            fixed = '{}_fixed'.format(i)
            test = '{}_test'.format(i)
            subg.node(fixed, '')
            subg.edge(fixed, test)

    #dott.render('FractalNetFancy.gv', view=True)

    frac = Digraph(comment='FractalNet', node_attr={'shape': 'box'})
    #frac.graph_attr['splines'] = 'ortho'
    #frac.graph_attr['rankdir'] = 'LR'xclip -sel clip < ~/.ssh/id_rsa.pub
    frac.edge_attr.update(arrowhead='vee')
    n_recs = 3
    frac.node('A', 'Gameboard', shape='box')
    frac.node('B', 'Action Map')
    rows = []
    for i in range(int(2**(n_recs-2))):
        i = (i + 1) * 2
        row = Digraph('child_{}'.format(i))
        row.attr(rank='same')
        row.attr(rankdir='LR')
        print(row)
        globals()['grp_{}'.format(i)] = row
        rows += [row]
    for i in range(n_recs):
        color = color_defaults[(n_recs - i) + (5-n_recs)]
        for j in range(2 ** i):
            n_j = n_recs - j
            j = j +1
            print(i, j)
            if not (i == n_recs - 1 and j % 2 == 1):
                row = globals()['grp_{}'.format(j * (2 ** (n_recs - i - 1)))]
            else:
                row = None
            print(row)
            fixed = '{}_{}'.format(i, j)
            if row is not None:
                row.node(fixed, '')
            else:
                frac.node(fixed, '')
            if i > 0 and j %2 == 0 and row is not None:
                row.edge('{}_{}'.format(i - 1, int(j/ 2)),fixed,
                         style='dashed',
                        arrowhead='none' ,
                       weight='-50'
                        )
            if j >1:
                frac.edge('{}_{}'.format(i, j-1), fixed, arrowhead='vee', color=color,
                        label='f_{' + str(i) + '-' + str(j - 0)+'}')
        if i == 0:
            frac.edge('1_2', 'B', arrowhead='none', style='dotted')
        frac.edge('A', '{}_{}'.format(i, 1), color=color,
     label='f_{' + str(i) + '-' + str(0)+'}')
    frac.node_attr.update(height='0.2')
    frac.node('E', 'S', shape='circle', style='dotted')
    frac.edge('1_2', 'E', style='dotted', arrowhead='none')



    for row in rows:
        if row is not None:
            frac.subgraph(row)

    frac.render('FractalNet.gv', view=True)

def create_gif(inputPath, outputPath, delay, finalDelay, loop):
	# grab all image paths in the input directory
	imagePaths = sorted(list(paths.list_images(inputPath)))
	# remove the last image path in the list
	lastPath = imagePaths[-1]
	imagePaths = imagePaths[:-1]
	# construct the image magick 'convert' command that will be used
	# generate our output GIF, giving a larger delay to the final
	# frame (if so desired)
	cmd = "convert -delay {} {} -delay {} {} -loop {} {}".format(
		delay, " ".join(imagePaths), finalDelay, lastPath, loop,
		outputPath)
	os.system(cmd)

def smooth_reward_curve(x, y):
    # Halfwidth of our smoothing convolution
    halfwidth = min(1000, int(np.ceil(len(x) / 30)))
    k = halfwidth
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='valid') / \
        np.convolve(np.ones_like(y), np.ones(2 * k + 1), mode='valid')
    downsample = max(int(np.floor(len(xsmoo) / 1e3)), 1)
    return xsmoo[::downsample], ysmoo[::downsample]


def fix_point(x, y, interval):
    np.insert(x, 0, 0)
    np.insert(y, 0, 0)

    fx, fy = [], []
    pointer = 0

    ninterval = int(max(x) / interval + 1)

    for i in range(ninterval):
        tmpx = interval * i

        while pointer + 1 < len(x) and tmpx > x[pointer + 1]:
            pointer += 1

        if pointer + 1 < len(x):
            alpha = (y[pointer + 1] - y[pointer]) / \
                (x[pointer + 1] - x[pointer])
            tmpy = y[pointer] + alpha * (tmpx - x[pointer])
            fx.append(tmpx)
            fy.append(tmpy)

    return fx, fy


def load_data(indir, smooth, bin_size, col=None, header='r', dots=False):
    datas = []
    if col is not None:
        infiles = glob.glob(os.path.join(indir, 'col_{}_eval.csv'.format(col)))
    else:
        infiles = glob.glob(os.path.join(indir, '*.monitor.csv'))
    if len(infiles) == 0:
        print('no files found at {}'.format(indir))

    for inf in infiles:
        with open(inf, 'r') as f:
            f.readline()
            f.readline()
            for line in f:
                tmp = line.split(',')
                t_time = float(tmp[2])
                header_idx = header_idxs[header]
                val = tmp[header_idx]
                tmp = [t_time, int(tmp[1]), float(val)]
                datas.append(tmp)

    datas = sorted(datas, key=lambda d_entry: d_entry[0])
    result = []
    timesteps = 0
    for i in range(len(datas)):
        result.append([timesteps, datas[i][-1]])
        timesteps += datas[i][1]

    if len(result) < bin_size:
        if len(result) > 2:
            bin_size = len(result) # hack, so we see graphs asap
        else:
            return [None, None]

    x, y = np.array(result)[:, 0], np.array(result)[:, 1]

    if smooth == 1:
        x, y = smooth_reward_curve(x, y)

    if smooth == 2:
        y = medfilt(y, kernel_size=9)

    if not dots:
        x, y = fix_point(x, y, bin_size)
    return [x, y]



class Plotter(object):
    def __init__(self, n_cols, indir, n_proc, max_steps=None):
        self.n_cols = n_cols + 1
        self.avgs = np.zeros((n_cols + 1))
        self.n_frames = np.zeros((n_cols + 1))
        self.n_samples = np.zeros((n_cols + 1)) # how many episodes per process,
        # this may be different for each column due to interrupted evaluation
        self.n_proc = n_proc # how many processes
        self.indir = indir
        self.max_steps = max_steps # this shouldn't change during frozen eval

        # keep our figures open and progressively animate new data
        self.evl_r_fig = None
        self.trn_r_fig = None
        self.trn_e_fig = None
        self.trn_p_fig = None


    def visdom_plot(self, viz, win, folder, game, name, num_steps, bin_size=100, smooth=1,
            n_graphs=None, x_lim=None, y_lim=None, man=False,
            eval=False, header='r', dots=False
            ):
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
        '''
         - n_graphs: specific to fractal columns
        '''
        if eval:
            if header == 'r':
                fig = self.evl_r_fig
        else:
            if header == 'r':
                fig = self.trn_r_fig
            elif header == 'e':
                fig = self.trn_e_fig
            elif header == 'p':
                fig = self.trn_p_fig
        if dots:
            smooth = 0

        if man:
            matplotlib.rcParams.update({'font.size': 14})
        if isinstance(folder, list):
            fld = folder
            folder = folder[0]
        else:
            fld = None
        if folder.endswith('logs'):
            evl = False
        elif folder.endswith('logs_eval'):
            evl = True
        if folder.endswith('logs'):
            evl = False
        elif folder.endswith('logs_eval'):
            evl = True
        if man:
            tick_fractions = np.array([1/4, 2/4, 3/4, 1])
        else:
            tick_fractions = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0])

        ticks = tick_fractions * num_steps
        tick_names = ["{:.0e}".format(tick) for tick in ticks]
        if man:
            tick_names[0] =''
            tick_names[2] = ''
            if not fig:
                fig = plt.figure(figsize=(5.6,5))
        else:
            if not fig:
                fig = plt.figure()

        if isinstance(fld, list):
            j = 0
            for f in fld:
                print(f)
                color = 0
                tx, ty = load_data(f, smooth, bin_size, col=-1, header=header)
                if tx is None or ty is None:
                    #print('could not find x y data columns in csv')
                    pass
                   #return win

                else:
                    if j == 0:
                        plt.plot(tx, ty, label="FullyConv", color=color_defaults[-1], linestyle='dashed')
                    else:
                        plt.plot(tx, ty, label="StrictlyConv", color=color_defaults[color])
                    color += 1
                j += 1
        elif n_graphs is not None:
            #print('indaplotter')
            color = 0
            for i in n_graphs:
                tx, ty = load_data(folder, smooth, bin_size, col=i, header=header)
                if tx is None or ty is None:
                    #print('could not find x y data columns in csv')
                    pass
                   #return win

                else:
                    plt.plot(tx, ty, label="col {}".format(i), color=color_defaults[color])
                    color += 1
        else:
            tx, ty = load_data(folder, smooth, bin_size, header=header, dots=dots)
            if tx is None or ty is None:
                return win
            if evl:
                color = 3
                plt.plot(tx, ty, label='det-eval', color=color_defaults[color])
            else:
                if dots:
                    plt.scatter(tx, ty, s=1, label='selected target')
                else:
                    plt.plot(tx, ty, label="non-det")


        if x_lim:
            plt.xlim(*x_lim)
        else:
            plt.xlim(0, num_steps * 1.01)
        if y_lim:
            plt.ylim(*y_lim)
        plt.xticks(ticks, tick_names)

        plt.xlabel('Number of Timesteps')
        header_name = header_names[header]
        plt.ylabel(header_name)
        plt.grid(b=True, which='both')
        plt.title('{}_{}'.format(game, header))
        if man:
            plt.legend(loc='upper left', bbox_to_anchor=(1,1))
            plt.tight_layout(w_pad=2)
        else:
            plt.legend(loc=4)
        if evl:
            figfolder = folder.replace('/logs_eval', '/eval_')
        else:
            figfolder = folder.replace('/logs', '/train_')
        print('should be saving graph now as {}'.format(figfolder))


        if man:
            figfile = './{}_{}_fig_man.png'.format(figfolder, header)
        else:
            figfile = './{}_{}_fig.png'.format(figfolder, header)
        plt.savefig(figfile, format='png')
        plt.show()
        plt.draw()

        image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
        plt.close(fig)



        # Show it in visdom
        image = np.transpose(image, (2, 0, 1))
        return viz.image(image, win=win)






    def get_col_avg(self, col=None):
        ''' Also records number of episodes '''
        if col is not None:
            infiles = glob.glob(os.path.join(self.indir, 'col_{}_eval.csv'.format(col)))
        else:
            infiles = glob.glob(os.path.join(self.indir, '*.monitor.csv'))
        if len(infiles) == 0:
            print('no files found at {}'.format(self.indir))

        i = 0
        net_reward = 0
        n_frames = 0
        for inf in infiles:
            with open(inf, 'r') as f:
                f.readline()
                f.readline()
                for line in f:
                    tmp = line.split(',')
                    r = float(tmp[0])
                    n_frames += float(tmp[1])
                    net_reward += r
                    i += 1
        if i != 0:
            avg_reward = net_reward / i
        else:
            avg_reward = 0
        self.avgs[col] = avg_reward
        self.n_frames[col] = n_frames
        return avg_reward

    def get_col_std(self, col=None):
        ''' We need to have already calculated avg for each col '''
        if col is not None:
            infiles = glob.glob(os.path.join(self.indir, 'col_{}_eval.csv'.format(col)))
        else:
            infiles = glob.glob(os.path.join(self.indir, '*.monitor.csv'))
        if len(infiles) == 0:
            print('no files found at {}'.format(self.indir))

        mean = self.avgs[col]
        i = 0
        net_deviation = 0
        for inf in infiles:
            with open(inf, 'r') as f:
                f.readline()
                f.readline()
                for line in f:
                    tmp = line.split(',')
                    r = float(tmp[0])
                    net_deviation += np.abs(mean - r)
                    i += 1
        if i != 0:
            avg_deviation = net_deviation / i
        else:
            avg_deviation = 0
        self.avgs[col] = avg_deviation
        return avg_deviation



    def bar_plot(self, viz, win, folder, game, name, num_steps, n_cols=None):
        import matplotlib.pyplot as plt
        plt.switch_backend('agg')
        fig = plt.figure()
        x = [i for i in range(-1, n_cols)]

        h = [self.get_col_avg(col = i) for i in range(-1, n_cols)]
        e = [self.get_col_std(col = i) for i in range(-1, n_cols)]
        plt.bar(x, h, yerr=e, color=color_defaults[:n_cols + 1])

        plt.xlabel('Columns')
        plt.ylabel('Rewards')
        for i, v in enumerate(h):
            plt.text(i - 1.25, v + 3, '{0:.3f}'.format(v))
            n_col_eps = self.n_frames[i] / self.max_steps # assuming max_steps does not change over course of evaluation
            plt.text(i - 1.25, v + 1, '{0:4.3e} eps.'.format(n_col_eps))
        plt.title(game)
        plt.legend(loc=4)
        figfolder = folder.replace('/logs_eval_', '/eval_')
       #figfolder = folder.replace('/logs', '/train_')
        print('should be saving graph now as {}'.format(figfolder))
        plt.savefig('{}/bar_fig.png'.format(figfolder), format='png')
        plt.show()
        plt.draw()

        image = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
        plt.close(fig)

        # Show it in visdom
        image = np.transpose(image, (2, 0, 1))
        return viz.image(image, win=win)

def man_eval_plot(indir, n_cols=5, num_steps=200000000, n_proc=20, x_lim=None, y_lim=None,
        title='', smooth=1):
    plotter = Plotter(n_cols=n_cols, indir=indir, n_proc=n_proc)

    from visdom import Visdom
    viz = Visdom()
    win = None
    if isinstance(indir, list):
        print('copy man\n')
        i = 0
        for d in indir:
            indir[i] = '{}/logs_eval'.format(d)
            i += 1
    else:
        indir = "{}/logs_eval".format(indir)
    win = plotter.visdom_plot(viz, win, indir, title,  "Fractal Net", num_steps=num_steps,
        n_graphs=range(-1,n_cols), x_lim=x_lim, y_lim=y_lim, man=True, bin_size=100, smooth=smooth)
    return win

if __name__ == "__main__":
    from visdom import Visdom
    import argparse
    viz = Visdom()
    win = None
    parser = argparse.ArgumentParser(description='viz')
    parser.add_argument('--load-dir', default=None,
            help='directory from which to load agent logs (default: ./trained_models/)')
    visdom_plot(viz, None, '/tmp/gym/', 'BreakOut', 'a2c', bin_size=100, smooth=1)
