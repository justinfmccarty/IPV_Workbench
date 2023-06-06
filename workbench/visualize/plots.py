import pandas as pd
import proplot as pplt
import numpy as np
from workbench.utilities import utils, circuits


def plot_curves(i_arrs, v_arrs, module_params, labels='label', colors='k', title=None, linewidth=1, linestyle='solid', fs=(5, 3),
                bypass=False, y_max=None, x_min=None, save=None, reverse=True, mpp=False):
    """
    Plot the IV curve from the I and V arrays. Currently only works for one module type
    :param i_arrs: current arrays (list of arrays or single array)
    :param v_arrs: voltage arrays (list of arrays or single array)
    :param module_params: a parameters dict for the module
    :param labels: list of labels that matches the length if the I and V array inputs
    :param colors: list of colors that matches the length if the I and V array inputs
    :param title:
    :param linewidth: list of linewidth that matches the length if the I and V array inputs
    :param linestyle: list of linestyle that matches the length if the I and V array inputs
    :param fs: size for the figure (tuple)
    :param bypass: whether to use the BYpass diode function
    :param y_max: set the plot y limit
    :param x_min: set the plot x limit
    :param save: either None or a file patht o save the figure
    :param reverse:
    :param mpp:
    :return:
    """
    # check for iterables
    if not isinstance(i_arrs, list):
        i_arrs = [i_arrs]
    if not isinstance(v_arrs, list):
        v_arrs = [v_arrs]
    if not isinstance(labels, list):
        labels = [labels] * len(i_arrs)
    if not isinstance(colors, list):
        colors = [colors] * len(i_arrs)
    if not isinstance(linestyle, list):
        linestyle = [linestyle] * len(i_arrs)
    if not isinstance(linewidth, list):
        linewidth = [linewidth] * len(i_arrs)

    if reverse == True:
        share_ = True
    else:
        share_ = False
    fig = pplt.figure(figsize=fs,
                      num="single_curve",
                      share=share_,
                      clear=True,
                      facecolor="white")
    ax1 = fig.add_subplot(1, 2, 1,
                          facecolor="white")
    ax2 = fig.add_subplot(1, 2, 2,
                          sharey=ax1,
                          facecolor="white")
    axes = [ax1, ax2]
    fig.suptitle(title)

    i_min = []
    i_max = []
    v_min = []
    v_max = []
    n = 0
    for i_arr, v_arr, label, lw, cl, ls in zip(i_arrs, v_arrs, labels, linewidth, colors, linestyle):
        arr = np.array([i_arr, v_arr])
        if bypass == True:
            V = circuits.apply_bypass_diode(arr[1, :], module_params)
            I = arr[0, :]
            df = pd.DataFrame({"i": I, "v": V})
        else:
            df = pd.DataFrame({"i": arr[0, :], "v": arr[1, :]})

        i_min.append(df['i'].min())
        i_max.append(df['i'].max())
        v_min.append(df['v'].min())
        v_max.append(df['v'].max())
        iv_left = df.plot('v', 'i', ax=axes[0], linewidth=lw, c=cl, linestyle=ls)
        iv_right = df.plot('v', 'i', ax=axes[1], label=label, linewidth=lw, c=cl, linestyle=ls)
        if mpp == True:
            mpp_ = circuits.find_mpp(arr)
            axes[1].scatter(mpp_[2], mpp_[1], c='red',alpha=0.5)
            if n + 1 == len(i_arrs):
                # solves the duplicated MPP in legend
                mpp_line_A = axes[1].hlines(y=mpp_[1],
                                            x1=0,
                                            # x2=np.max(arr[1]) * 10,
                                            x2=mpp_[2],
                                            color='red', alpha=0.5,
                                            label='MPP',
                                            linewidth=0.5, linestyle="dashed")
            else:
                mpp_line_A = axes[1].hlines(y=mpp_[1],
                                            x1=0,
                                            x2=mpp_[2],
                                            # x2=np.max(arr[1]) * 10,
                                            color='red', alpha=0.5,
                                            linewidth=0.5, linestyle="dashed")

            mpp_line_V = axes[1].vlines(x=mpp_[2],
                                        y1=0,
                                        y2=mpp_[1],
                                        # y2=np.max(arr[0]) * 10,
                                        color='red', alpha=0.5,
                                        linewidth=0.5, linestyle="dashed")
            n += 1


    if np.min(v_min) * 1.25 == 0:
        axes[0].set_xlim([module_params['diode_threshold'] * 2, 0])
    else:
        axes[0].set_xlim(right=0, left=np.min(v_min) * 1.25)
    if y_max == None:
        axes[0].set_ylim([0, np.max(i_max)])
    else:
        axes[0].set_ylim([0, y_max])

    if x_min == None:
        x_min = np.min(v_min) * 1.25

    axes[0].set_xlim([x_min, 0])
    axes[1].set_xlim([0, np.max(v_max) * 1.05])

    axes[0].legend().set_visible(False)
    axes[1].legend().set_visible(False)

    axes[0].set_ylabel('current [A]')
    axes[0].set_xlabel('voltage [V]')
    axes[1].set_xlabel('voltage [V]')
    if bypass == True:
        bypass_lw = 3
        bypass_ls = 'dashed'
        bypass_color = 'grey'
        axes[0].vlines(-0.5, 0, 100,
                       linewidth=bypass_lw,
                       linestyle=bypass_ls,
                       color=bypass_color)
        # necessary for legend
        if reverse == True:
            axes[1].vlines(x=1,
                           y1=0,
                           y2=0,
                           label='Bypass Diode',
                           linewidth=bypass_lw,
                           linestyle=bypass_ls,
                           color=bypass_color)
    if labels[0] is None:
        pass
    else:
        axes[1].legend(loc="upper right", facecolor="lightgrey",
                       # bbox_to_anchor=(1.02, 1),
                       ncol=1,
                       fancybox=False, shadow=False,
                       borderaxespad=1,
                       prop={'size': 6})

    if reverse == False:
        fig.delaxes(axes[0])
        axes[1].set_ylabel('current [A]')
        fig.suptitle("")
        axes[1].format(title=title, titleweight='bold')
    if save != None:
        fig.savefig(save, dpi=300, bbox_inches='tight')

    pplt.show()
    fig.clear()
    pplt.close(fig)

