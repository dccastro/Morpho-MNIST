import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker


# def plot_radial_hist(angles, ax=None, **kwargs):
#     if ax is None:
#         ax = plt.gca()
#     lims = [-np.pi / 4, np.pi / 4]
#     bin_width = np.pi / 60
#     bins = np.arange(lims[0], lims[1] + bin_width, bin_width) - bin_width / 2
#     counts, bins, patches = ax.hist(angles, bins=bins, **kwargs)
#     ax.set_theta_zero_location('N')
#     ax.set_theta_direction(-1)
#     ax.set_thetalim(lims)
#     ax.set_rorigin(-ax.get_rmax())
#     ax.set_xticks(np.arange(lims[0] + 1e-3, lims[1] + 2e-3, np.pi / 12))
#     colour_hist(patches, counts, plt.cm.Blues)


# def colour_hist(bars, counts, cmap):
#     for count, bar in zip(counts, bars):
#         bar.set_facecolor(cmap(count / max(counts)))


if __name__ == '__main__':
    df = pd.read_csv("../mnist_features_x8_test.csv", index_col='index')
    df['slant'] = np.rad2deg(np.arctan(-df['shear']))

    cols = ['mean_thck', 'slant', 'width', 'height']
    labels = ['Thickness', 'Slant', 'Width', 'Height']
    fig, axs = plt.subplots(2, len(cols), sharex='col', sharey='row', figsize=(12, 4),
                            gridspec_kw=dict(height_ratios=[10, 1], hspace=.1, wspace=.1, left=0,
                                             right=1))

    def format_violinplot(parts):
        for pc in parts['bodies']:
            pc.set_facecolor('#1f77b480')
            pc.set_edgecolor('C0')
            pc.set_alpha(None)

    for c, col in enumerate(cols):
        ax = axs[0, c]
        parts = ax.violinplot([df.loc[df['digit'] == d, col] for d in range(10)],
                              positions=np.arange(10), vert=False, widths=.7,
                              showextrema=False, showmedians=True)
        format_violinplot(parts)
        format_violinplot(axs[1, c].violinplot(df[col], vert=False, widths=.7,
                                               showextrema=False, showmedians=True))
        ax.set_title(labels[c])
        ax.set_axisbelow(True)
        ax.grid(axis='x')
        axs[1, c].set_axisbelow(True)
        axs[1, c].grid(axis='x')

    axs[0, 0].yaxis.set_major_locator(ticker.MultipleLocator(1))
    axs[0, 0].set_ylabel("Digit")
    axs[1, 0].set_yticks([1])
    axs[1, 0].set_yticklabels(["All"])
    axs[1, 0].set_ylim(.5, 1.5)
    for ax in axs[:, 1]:
        ax.axvline(0., lw=1., ls=':', c='k')
    axs[1, 1].set_xlim(-46, 46)
    axs[1, 1].xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.0f}$\degree$"))
    plt.savefig("../fig/distributions_x8_test.pdf", bbox_inches='tight')
    plt.show()
