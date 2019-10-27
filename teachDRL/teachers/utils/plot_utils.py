# Plotting functions
# WARNING ugly code, sorry

import matplotlib.patches as patches
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.colorbar as cbar
import matplotlib.pyplot as plt
import imageio
import numpy as np
import copy
import os
from matplotlib.patches import Ellipse
import matplotlib.colors as colors


# scale numpy 1d array to [-1:1] given its bounds
# bounds must be of the form [[min1,max1],[min2,max2],...]
def scale_vector(values, bounds):
    mins_maxs_diff =  np.diff(bounds).squeeze()
    mins = bounds[:, 0]
    return (((values - mins) * 2) / mins_maxs_diff) - 1

def unscale_vector(scaled_values, bounds=[[-1,1]]):
    mins_maxs_diff =  np.diff(bounds).squeeze()
    mins = bounds[:, 0]
    return (((scaled_values + 1) * mins_maxs_diff) / 2) + mins

def plt_2_rgb(ax):
    ax.figure.canvas.draw()
    data = np.frombuffer(ax.figure.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(ax.figure.canvas.get_width_height()[::-1] + (3,))
    return data

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def scatter_plot(data, ax=None, emph_data=None, xlabel='stump height', ylabel='spacing', xlim=None, ylim=None):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(7, 7))
    Xs, Ys = [d[0] for d in data], [d[1] for d in data]
    if emph_data is not None:
        emphXs, emphYs = [d[0] for d in emph_data], [d[1] for d in emph_data]
    ax.plot(Xs, Ys, 'r.', markersize=2)
    ax.axis('equal')
    ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.set_ylim(bottom=ylim[0], top=ylim[1])
    if emph_data is not None:
        ax.plot(emphXs, emphYs, 'b.', markersize=5)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)

def plot_regions(boxes, interests, ax=None, xlabel='', ylabel='', xlim=None, ylim=None, bar=True):
    ft_off = 15
    # Create figure and axes
    if ax == None:
        f, ax = plt.subplots(1, 1, figsize=(8, 7))
    # Add the patch to the Axes
    #print(boxes)
    for b, ints in zip(boxes, interests):
        # print(b)
        lx, ly = b.low
        hx, hy = b.high
        c = plt.cm.jet(ints)
        rect = patches.Rectangle([lx, ly], (hx - lx), (hy - ly), linewidth=3, edgecolor='white', facecolor=c)
        ax.add_patch(rect)
        # plt.Rectangle([lx,ly],(hx - lx), (hy - ly))

    if bar:
        cax, _ = cbar.make_axes(ax, shrink=0.8)
        cb = cbar.ColorbarBase(cax, cmap=plt.cm.jet)
        cb.set_label('Absolute Learning Progress', fontsize=ft_off + 5)
        cax.tick_params(labelsize=ft_off + 0)
    ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.set_ylim(bottom=ylim[0], top=ylim[1])
    ax.set_xlabel(xlabel, fontsize=ft_off + 0)
    ax.set_ylabel(ylabel, fontsize=ft_off + 0)
    ax.tick_params(axis='both', which='major', labelsize=ft_off + 5)
    ax.set_aspect('equal', 'box')

def region_plot_gif(all_boxes, interests, iterations, goals, gifname='riac', rewards=None, ep_len=None,
                    gifdir='graphics/', xlim=[0,1], ylim=[0,1], scatter=False, fs=(9,6), plot_step=250):
    gifdir = 'graphics/' + gifdir
    ft_off = 15
    plt.ioff()
    print("Making an exploration GIF: " + gifname)
    # Create target Directory if don't exist
    tmpdir = 'tmp/'
    tmppath = gifdir + 'tmp/'
    if not os.path.exists(gifdir):
        os.mkdir(gifdir)
        print("Directory ", gifdir, " Created ")
    if not os.path.exists(tmppath):
        os.mkdir(tmppath)
        print("Directory ", tmppath, " Created ")
    filenames = []
    images = []
    steps = []
    mean_rewards = []
    for i in range(len(goals)):
        if i > 0 and (i % plot_step == 0):
            f, (ax0) = plt.subplots(1, 1, figsize=fs)
            ax = [ax0]
            if scatter:
                scatter_plot(goals[0:i], ax=ax[0], emph_data=goals[i - plot_step:i], xlim=xlim, ylim=ylim)
            idx = 0
            cur_idx = 0
            for j in range(len(all_boxes)):
                if iterations[j] > i:
                    break
                else:
                    cur_idx = j
            plot_regions(all_boxes[cur_idx], interests[cur_idx], ax=ax[0], xlim=xlim, ylim=ylim)

            f_name = gifdir+tmpdir+"scatter_{}.png".format(i)
            plt.suptitle('Episode {}'.format(i), fontsize=ft_off+0)
            images.append(plt_2_rgb(plt.gca()))
            plt.close(f)
    imageio.mimsave(gifdir + gifname + '.gif', images, duration=0.4)


def draw_ellipse(position, covariance, ax=None, color=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()

    covariance = covariance[0:2,0:2]
    position = position[0:2]

    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)

    # Draw the Ellipse
    for nsig in range(2, 3):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs, color=color))

def draw_competence_grid(ax, comp_grid, x_bnds, y_bnds, bar=True):
    comp_grid[comp_grid == 100] = 1000
    ax.pcolor(x_bnds, y_bnds, np.transpose(comp_grid),cmap=plt.cm.gray, edgecolors='k', linewidths=2,
              alpha=0.3)
    if bar:
        cax, _ = cbar.make_axes(ax,location='left')
        cb = cbar.ColorbarBase(cax, cmap=plt.cm.gray)
        cb.set_label('Competence')
        cax.yaxis.set_ticks_position('left')
        cax.yaxis.set_label_position('left')

def plot_gmm(weights, means, covariances, X=None, ax=None, xlim=[0,1], ylim=[0,1], xlabel='', ylabel='',
             bar=True, bar_side='right',no_y=False, color=None):
    ft_off = 15

    ax = ax or plt.gca()
    cmap = truncate_colormap(plt.cm.autumn_r, minval=0.2,maxval=1.0)
    #colors = [plt.cm.jet(i) for i in X[:, -1]]
    if X is not None:
        colors = [cmap(i) for i in X[:, -1]]
        sizes = [5+np.interp(i,[0,1],[0,10]) for i in X[:, -1]]
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=sizes, zorder=2)
        #ax.axis('equal')
    w_factor = 0.6 / weights.max()
    for pos, covar, w in zip(means, covariances, weights):
        draw_ellipse(pos, covar, alpha=0.6, ax=ax, color=color)

    #plt.margins(0, 0)
    ax.set_xlim(left=xlim[0], right=xlim[1])
    ax.set_ylim(bottom=ylim[0], top=ylim[1])
    if bar:
        cax, _ = cbar.make_axes(ax, location=bar_side, shrink=0.8)
        cb = cbar.ColorbarBase(cax, cmap=cmap)
        cb.set_label('Absolute Learning Progress', fontsize=ft_off + 5)
        cax.tick_params(labelsize=ft_off + 0)
        cax.yaxis.set_ticks_position(bar_side)
        cax.yaxis.set_label_position(bar_side)
    #ax.yaxis.tick_right()
    if no_y:
        ax.set_yticks([])
    else:
        ax.set_ylabel(ylabel, fontsize=ft_off + 5)
        #ax.yaxis.set_label_position("right")
    ax.set_xlabel(xlabel, fontsize=ft_off + 5)
    ax.tick_params(axis='both', which='major', labelsize=ft_off + 5)
    ax.set_aspect('equal', 'box')

def gmm_plot_gif(bk, gifname='test', gifdir='graphics/', ax=None,
                 xlim=[0,1], ylim=[0,1], fig_size=(9,6), save_imgs=False, title=True, bar=True):
    gifdir = 'graphics/' + gifdir
    plt.ioff()
    # Create target Directory if don't exist
    tmpdir = 'tmp/'
    tmppath = gifdir + 'tmp/'
    if not os.path.exists(gifdir):
        os.mkdir(gifdir)
        print("Directory ", gifdir, " Created ")
    if not os.path.exists(tmppath):
        os.mkdir(tmppath)
        print("Directory ", tmppath, " Created ")
    print("Making " + tmppath + gifname + ".gif")
    images = []
    old_ep = 0
    gen_size = int(len(bk['tasks_lps']) / len(bk['episodes']))
    gs_lps = bk['tasks_lps']
    for i,(ws, covs, means, ep) in enumerate(zip(bk['weights'], bk['covariances'], bk['means'], bk['episodes'])):
            plt.figure(figsize=fig_size)
            ax = plt.gca()
            plot_gmm(ws, means, covs, np.array(gs_lps[old_ep + gen_size:ep + gen_size]),
                     ax=ax, xlim=xlim, ylim=ylim,
                     bar=bar)  # add gen_size to have gmm + the points that they generated, not they fitted
            if 'comp_grids' in bk:  # add competence grid info
                draw_competence_grid(ax,bk['comp_grids'][i], bk['comp_xs'][i], bk['comp_ys'][i])
            if 'start_points' in bk:  # add lineworld info
                draw_lineworld_info(ax, bk['start_points'], bk['end_points'], bk['current_states'][i])
            f_name = gifdir+tmpdir+gifname+"_{}.png".format(ep)
            if title:
                plt.suptitle('Episode {} | nb Gaussians:{}'.format(ep,len(means)), fontsize=20)
            old_ep = ep
            if save_imgs: plt.savefig(f_name, bbox_inches='tight')
            images.append(plt_2_rgb(ax))
            plt.close()
    imageio.mimsave(gifdir + gifname + '.gif', images, duration=0.4)

def random_plot_gif(bk, step=250, gifname='test', gifdir='graphics/', ax=None,
                 xlim=[0,1], ylim=[0,1], fig_size=(9,6), save_imgs=False, title=True, bar=True):
    gifdir = 'graphics/' + gifdir
    plt.ioff()
    # Create target Directory if don't exist
    tmpdir = 'tmp/'
    tmppath = gifdir + 'tmp/'
    if not os.path.exists(gifdir):
        os.mkdir(gifdir)
        print("Directory ", gifdir, " Created ")
    if not os.path.exists(tmppath):
        os.mkdir(tmppath)
        print("Directory ", tmppath, " Created ")
    print("Making " + tmppath + gifname + ".gif")
    images = []
    tasks = np.array(bk['tasks'])
    for i,(c_grids, c_xs, c_ys) in enumerate(zip(bk['comp_grids'], bk['comp_xs'], bk['comp_ys'])):
            plt.figure(figsize=fig_size)
            ax = plt.gca()
            draw_competence_grid(ax, c_grids, c_xs, c_ys)
            ax.scatter(tasks[i*step:(i+1)*step, 0], tasks[i*step:(i+1)*step, 1], c='blue', s=2, zorder=2)

            ax.set_xlim(left=xlim[0], right=xlim[1])
            ax.set_ylim(bottom=ylim[0], top=ylim[1])
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.set_aspect('equal', 'box')
            
            f_name = gifdir+tmpdir+gifname+"_{}.png".format(i)
            if title:
                plt.suptitle('Episode {}'.format(i*step), fontsize=20)
            if save_imgs: plt.savefig(f_name, bbox_inches='tight')
            images.append(plt_2_rgb(ax))
            plt.close()

    imageio.mimsave(gifdir + gifname + '.gif', images, duration=0.4)