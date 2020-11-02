import numpy as np
import pandas as pd
import matplotlib.cm as cm

egroups = {
    'left_front' : [11, 12, 13, 14],
    'right_front' : [18, 17, 16, 15],
    'left_front_middle' : [9, 10],
    'right_front_middle' : [20, 19],
    'left_back_middle' : [3, 4, 6, 7],
    'right_back_middle' : [26, 25, 23, 22],
    'left_back_middle_center' : [5, 8],
    'right_back_middle_center' : [24, 21],
    'left_back' : [1, 2],
    'right_back' : [28, 27],
    'left_bottom' : [0],
    'right_bottom' : [29],
}

def _process_exp(exp):
    exp.ch_coordinates['z'] = 0
    exp.ch_coordinates['group'] = ''
    exp.ch_coordinates['gid'] = 0
    exp.ch_coordinates['wgid'] = 0
    for i, (g, idx) in enumerate(egroups.items()):
        exp.ch_coordinates.loc[idx, 'group'] = g
        exp.ch_coordinates.loc[idx, 'gid'] = i
        exp.ch_coordinates.loc[sorted(idx), 'wgid'] = idx
    exp.ch_coordinates = exp.ch_coordinates.sort_values(['gid', 'wgid'])
    exp.ch_coordinates['order'] = 0
    _left = exp.ch_coordinates.index[exp.ch_coordinates.group.str.contains('left')]
    exp.ch_coordinates.loc[_left, 'order'] = range(len(_left))
    _right = exp.ch_coordinates.index[exp.ch_coordinates.group.str.contains('right')]
    exp.ch_coordinates.loc[_right, 'order'] = len(_left)+np.arange(len(_right))[::-1]
    exp.ch_coordinates.sort_index(inplace=True)
    exp.ch_coordinates.drop('wgid', inplace=True, axis=1)
    exp._ch_coordinates_processed = True
    return exp

# define a function to quickly plot the electrode map with or without borders
def plot_electrode_map(ax, highlight=None, labels=False, numbers=True, box=False, cmap=cm.Paired, s=50):
    metaelectrodes = pd.read_pickle('metaelectrodes.pkl')
    colors = np.array(metaelectrodes.gid.map(lambda x: cmap(x/11, 0.9)))
    if highlight in set(metaelectrodes.group):
        colors = np.array(
            metaelectrodes.apply(
                lambda row: cmap(row.gid/12, 0.9) if row.group==highlight else cm.Greys(0.5,0.5), axis=1
            )
        )
    metaelectrodes.plot(
        kind='scatter', x='ML', y='AP', marker='o', ax=ax, legend=False, c=colors, s=s
    )
    if numbers:
        for i in metaelectrodes.index:
            ax.annotate(
                metaelectrodes.index[i], metaelectrodes.loc[i, ['AP', 'ML']][::-1]+[0, 0.2],
                xycoords='data', ha='center'
            )
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 3)
    if labels:
        ax.set_title('Electrode map')
    if not labels:
        ax.set_xlabel('')
        ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    if not box:
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
    return

# function to show electrode groups along an axis instead of electrode numbers
def draw_groups(ax, cmap=cm.Paired):
    ax.set_xlim(-0.5, 29.5)
    ax.set_ylim(29.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    nt = 0
    def add_patch(df):
        nonlocal nt, ax
        ax.add_patch(patches.Rectangle(
            (nt+0.05-0.5, 30), len(df)-0.05, 1, clip_on=False,
            color=cmap(df.gid.iloc[0]/12, 0.9), label=df.group.iloc[0]
        ))
        ax.add_patch(patches.Rectangle(
            (-2, nt+0.05-0.5), 1, len(df), clip_on=False, color=cmap(df.gid.iloc[0]/12, 0.9)
        ))
        nt += len(df)
    metaelectrodes.sort_values('order').groupby('gid', sort=False).apply(add_patch)
    return