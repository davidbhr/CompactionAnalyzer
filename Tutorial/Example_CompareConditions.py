"""Compare horizontal, crossing and vertical synthetic fibers (no biological data).

Run from the repository root:
    python Tutorial/Example_CompareConditions.py --output comparison_output
Add --saenopy to also save measurements for the GUI's Data Analysis tab.
"""
from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage.draw import line_aa

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from CompactionAnalyzer import analyze_image


METRICS = ['mean_alignment_line', 'mean_deviation_line', 'mean_coherency']
SIGMA_TENSOR = 10  # pixels; the same neighborhood size for every condition
EDGE = 15  # pixels
CONDITIONS = [('Horizontal fibers', 'horizontal', 20, '#258172'),
              ('Crossing fibers', 'crossing', 120, '#8865ad'),
              ('Vertical fibers', 'vertical', 220, '#3d78a5')]


def make_image(direction, seed):
    """Make fibers with a preferred direction or crossings in many directions.

    Only the direction distribution changes between conditions. Fiber count,
    lengths, thickness, image noise, selected area and reference stay the same.
    The seed reproduces an image; each group uses three independent seeds.
    """
    if direction not in ('horizontal', 'crossing', 'vertical'):
        raise ValueError('Choose horizontal, crossing or vertical fibers.')
    rng = np.random.default_rng(seed)
    image = np.zeros((512, 512))
    for _ in range(350):
        x, y = rng.uniform(-60, 572, 2)
        length = rng.uniform(60, 140)
        if direction == 'crossing':
            angle = rng.uniform(0, np.pi)
        else:
            preferred_angle = 0 if direction == 'horizontal' else 90
            angle = np.deg2rad(preferred_angle + rng.normal(0, 8))
        dx, dy = length / 2 * np.cos(angle), length / 2 * np.sin(angle)
        rows, columns, weights = line_aa(
            round(y - dy), round(x - dx), round(y + dy), round(x + dx))
        inside = (rows >= 0) & (rows < 512) & (columns >= 0) & (columns < 512)
        image[rows[inside], columns[inside]] += weights[inside] * rng.uniform(.7, 1.3)

    image = gaussian_filter(image, 1.2)
    image /= np.percentile(image, 99.5)
    image = np.clip(1500 + 50000 * image + rng.normal(0, 650, image.shape),
                    0, 65535).astype(np.uint16)
    y, x = np.indices(image.shape)
    mask = ((x - 255.5) / 205) ** 2 + ((y - 255.5) / 165) ** 2 <= 1
    reference_line = ((80, 256), (432, 256))  # (x, y), horizontal in every image
    return image, mask, reference_line


def run(output, saenopy=False):
    """Return one row per image and group means, SEMs and image counts."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, examples = [], []
    for condition, direction, first_seed, _ in CONDITIONS:
        for index in range(3):
            image, mask, line = make_image(direction, seed=first_seed + index)
            values = analyze_image(
                image, mode='line_orientation', reference_line=line, mask=mask,
                sigma_tensor=SIGMA_TENSOR, edge=EDGE, outputs=METRICS,
            )
            rows.append(dict(condition=condition, image=index,
                             seed=first_seed + index, **values))
            if index == 0:
                examples.append((condition, image, mask, line))
            if saenopy:
                folder = output / condition.lower().replace(' ', '_')
                save_measurement(image, mask, line, folder, index)

    per_image = pd.DataFrame(rows)
    # First average within each image (above), then across images (here).
    # Each image has equal weight, regardless of its number of selected pixels.
    summary = per_image.groupby('condition', sort=False)[METRICS].agg(
        ['mean', 'sem', 'count'])
    per_image.to_csv(output / 'per_image.csv', index=False)
    summary.to_csv(output / 'group_summary.csv')
    save_figure(per_image, examples, output / 'comparison.png')
    print(summary.to_string())
    return per_image, summary


def save_measurement(image, mask, line, folder, index):
    """Optional GUI export; image coordinates use one synthetic unit per pixel."""
    import imageio.v2 as imageio
    from saenopy.gui.orientation.modules.result import ResultOrientation
    from saenopy.gui.orientation.modules.analysis import analyze_orientation

    folder.mkdir(parents=True, exist_ok=True)
    image_name, mask_name = f'image_{index}.tif', f'mask_{index}.tif'
    imageio.imwrite(folder / image_name, image)
    imageio.imwrite(folder / mask_name, mask.astype(np.uint8) * 255)
    result = ResultOrientation(
        output=str(folder / f'image_{index}.saenopyOrientation'),
        image_fiber=image_name, image_cell=mask_name,
        pixel_size=1., analysis_mode='line_orientation')
    result.shape = image.shape
    parameters = dict(edge=EDGE, sigma_tensor=SIGMA_TENSOR, sigma_tensor_type='pixel',
                      sigma_first_blur=.5, SaveNumpy=False, plotting=False)
    parameters.update(zip(('line_x1', 'line_y1', 'line_x2', 'line_y2'),
                          np.asarray(line).ravel()))
    analyze_orientation(result, parameters, dict(
        mask_sigma=0., mask_threshold=.5, mask_invert=False))


def save_figure(per_image, examples, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = [color for _, _, _, color in CONDITIONS]
    fig = plt.figure(figsize=(11.6, 7.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 6, height_ratios=[1.15, 1])
    image_axes = [fig.add_subplot(grid[0, start:start + 2]) for start in (0, 2, 4)]
    plot_axes = [fig.add_subplot(grid[1, :3]), fig.add_subplot(grid[1, 3:])]
    for ax, (condition, image, mask, line), color in zip(image_axes, examples, colors):
        ax.imshow(image, cmap='gray', vmin=1500, vmax=58500)
        ax.contour(mask, levels=[.5], colors=[color], linewidths=1)
        ax.axline(*line, color='#ffe54d', ls='--', lw=1)
        ax.set(xlim=(-.5, 511.5), ylim=(511.5, -.5), title=condition)
        ax.set_axis_off()
    plots = [('mean_alignment_line', 'Alignment to line', (-1, 1)),
             ('mean_coherency', 'Mean local coherency', (0, 1))]
    for ax, (metric, title, limits) in zip(plot_axes, plots):
        ax.axhline(0, color='#b0b0b0', lw=.8, zorder=0)
        for position, ((condition, rows), color) in enumerate(zip(
                per_image.groupby('condition', sort=False), colors)):
            values = rows[metric]
            ax.bar(position, values.mean(), color=color, alpha=.3, width=.6)
            ax.errorbar(position, values.mean(), yerr=values.sem(),
                        color='black', capsize=4, fmt='_', markersize=12)
            ax.scatter(position + np.linspace(-.13, .13, len(values)), values,
                       color=color, s=22, zorder=3)
        ax.set(title=title, ylim=limits, xticks=[0, 1, 2],
               xticklabels=['Horizontal', 'Crossing', 'Vertical'])
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Synthetic conditions · 3 images per group · points: image means · bars: group mean ± SEM',
                 fontsize=11)
    fig.savefig(output, dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--saenopy', action='store_true')
    args = parser.parse_args()
    run(args.output, args.saenopy)
