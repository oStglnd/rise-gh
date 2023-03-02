
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# define paths
home_path = os.getcwd()
data_path = home_path + '\\data\\climate_merged.csv'
plot_path = home_path + '\\plots\\climate\\'

# define HYPERPARAMS
plot_unit   = '3 day'
plot_suffix = '3d'
plot_nsteps = 8640

# define categories for plots
plot_cat = [
    ('temp', 'GH_avg'),
    ('temp', 'GH'),
    ('temp', 'DC'),
    ('temp', 'OUT'),
    ('humidity', 'GH_avg'),
    ('humidity', 'GH'),
    ('humidity', 'DC'),
    ('humidity', 'OUT'),
    ('pressure', 'Default'),
    ('pressure', 'Differential'),
    ('pressure', 'OUT')
]

# get data
data = pd.read_csv(
    data_path,
    header=[0, 1, 2],
    index_col=0
)
data.index = pd.to_datetime(data.index.values)

# set plot setting
sns.set_theme()
sns.set_style('white')

# define dict/mapping for palettes
palettes = [
    'YlOrRd_r',
    'YlOrRd_r',
    'YlOrRd_r',
    'YlOrRd_r',
    'crest',
    'crest',
    'crest',
    'crest',
    'magma',
    'magma',
    'magma',
    # 'mako'
]
palettes = dict(zip(plot_cat, palettes))

# define dict/mapping for UNITS (y label)
units = [
    '$^\circ$C',
    '$^\circ$C',
    '$^\circ$C',
    '$^\circ$C',
    '$\%$',
    '$\%$',
    '$\%$',
    '$\%$',
    'Pa',
    'Pa',
    'hPa'
    # 'l/s',
]
units = dict(zip(plot_cat, units))

# iterate over categories and save figs
for cat in plot_cat:

    ax = sns.relplot(
        data=data.iloc[-plot_nsteps:][cat],
        kind='line',
        palette = palettes[cat],
        dashes=False,
        markers=False,
        legend='full'
    )

    ax.set_xticklabels(step=2)
    
    plt.yticks(rotation=45)
    plt.xlabel('Timestamp')
    plt.ylabel(units[cat], loc='center', rotation=0, fontsize=12, labelpad=30)
    plt.title(cat[0].title() + ' - ' + cat[1].upper() + ', ' + plot_unit)
    fpath = plot_path + cat[0] + '_' + cat[-1] + '_' + plot_suffix + '.png'
    plt.savefig(fpath, bbox_inches='tight', dpi=1000)