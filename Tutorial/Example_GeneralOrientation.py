"""X/Y orientation in collagen or phalloidin images, with an optional phalloidin mask.

The whole-image figure compares collagen with independent Gaussian pixel noise.
Both use the same image dimensions, tensor window, initial blur and edge crop.

python Tutorial/Example_GeneralOrientation.py --case phalloidin --figure phalloidin.png
python Tutorial/Example_GeneralOrientation.py --case whole --figure whole.png
python Tutorial/Example_GeneralOrientation.py --case mask --figure mask.png
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CompactionAnalyzer import analyze_image


def analyze_noise_reference(shape, scale, sigma_tensor, edge, count=8, seed=100):
    """Average independent noise fields, with no imposed preferred direction.

    Seeds are fixed for reproducibility. Each image's histogram sums to one
    before averaging. The first image is shown; the curve uses all images.
    Pixel noise is not smoothed before the common analysis pipeline.
    """
    import numpy as np

    if count < 2:
        raise ValueError('Use at least two noise images to calculate a SEM.')
    bins = np.arange(-90, 91, 5)
    histograms, measurements = [], []
    example = {}
    for index in range(count):
        image = np.random.default_rng(seed + index).normal(size=shape)
        values = analyze_image(
            image, mode='axis_orientation_only', scale=scale,
            sigma_tensor=sigma_tensor, edge=edge,
            outputs=['angle_x', 'mean_alignment_x', 'mean_deviation_x', 'mean_coherency'],
        )
        angles = values['angle_x']
        counts, _ = np.histogram(angles[np.isfinite(angles)], bins=bins)
        histograms.append(counts / counts.sum())
        measurements.append({key: value for key, value in values.items()
                             if key.startswith('mean_')})
        if index == 0:
            example = {'image': image, 'angle_x': angles}
    return dict(**example, bins=bins, histograms=np.asarray(histograms),
                per_image=measurements, count=count,
                **{key: float(np.mean([row[key] for row in measurements]))
                   for key in measurements[0]})


def run(case, figure=None):
    data = ROOT/'Tutorial'
    if case == 'whole':
        source = sorted((data/'EmptyGel/Pos01').glob('*C003*.tif'))[0]
        scale, edge, sigma_um = .249, 40, 7.
        mode, parameters = 'axis_orientation_only', {}
    elif case in ('phalloidin', 'mask'):
        source = sorted((data/'StressFibers/controlzoom1.5').glob('*.tif'))[0]
        scale, edge, sigma_um = .48, 10, 3.
        mode, parameters = ('axis_orientation_masked', {'mask_threshold': .25}) if case == 'mask' else ('axis_orientation_only', {})
    else:
        raise ValueError("case must be 'whole', 'phalloidin' or 'mask'")
    result = analyze_image(
        source, mode=mode, scale=scale, sigma_tensor=sigma_um/scale, edge=edge,
        outputs=['angle_x', 'coherency', 'selection_mask', 'mean_deviation_x', 'mean_alignment_x', 'mean_coherency'],
        **parameters,
    )
    print(case, {name: value for name, value in result.items() if name.startswith('mean_')})
    if figure is not None:
        save_figure(source, result, case, scale, edge, figure, sigma_tensor=sigma_um/scale)
    return result


def save_figure(source, result, case, scale, edge, output, noise_reference=None, sigma_tensor=None):
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    raw = imageio.imread(source)
    if raw.ndim == 3:
        from skimage.color import rgb2gray
        raw = rgb2gray(raw)
    if case == 'whole':
        if noise_reference is None:
            noise_reference = analyze_noise_reference(
                raw.shape, scale, 7/scale if sigma_tensor is None else sigma_tensor, edge)
        print('noise reference', {key: value for key, value in noise_reference.items()
                                  if key.startswith('mean_')})
        return save_noise_comparison(raw, result, noise_reference, scale, edge, output)
    cropped = raw[edge:-edge, edge:-edge] if edge else raw
    low, high = np.percentile(cropped, [1, 99])
    panels = 3
    fig, axes = plt.subplots(1, panels, figsize=(4.2 * panels, 4.4), constrained_layout=True)
    for ax in axes:
        ax.imshow(cropped, cmap='gray', vmin=low, vmax=high)
        ax.set_axis_off()
    if case == 'mask':
        overlay = np.zeros((*result['selection_mask'].shape, 4))
        overlay[result['selection_mask']] = [1., .2, .2, .28]
        axes[0].imshow(overlay)
    overlay_alpha = .5  # Keep the underlying phalloidin structures visible.
    mapped = axes[1].imshow(result['angle_x'], cmap='gist_rainbow', vmin=-90, vmax=90, alpha=overlay_alpha)
    if case in ('phalloidin', 'mask'):
        # Coherency measures local directional consistency independently of X/Y.
        coherent = axes[2].imshow(result['coherency'], cmap='viridis', vmin=0, vmax=1, alpha=overlay_alpha)
        axes[2].set_title('Coherency', fontsize=12)
        coherence_bar = fig.colorbar(coherent, ax=axes[2], fraction=.045, pad=.02,
                                    ticks=[0, .5, 1])
        coherence_bar.set_label('Coherency (0–1)', fontsize=10)
    for ax in axes:
        ax.plot([25, 25+50/scale], [cropped.shape[0]-30]*2, color='white', lw=3)
        ax.text(25, cropped.shape[0]-43, '50 µm', color='white', fontsize=9)
    axes[0].set_title({'whole': 'Collagen', 'phalloidin': 'Phalloidin', 'mask': 'Phalloidin + selected area'}[case], fontsize=12)
    axes[1].set_title('Angle to X axis within mask' if case == 'mask' else 'Angle to X axis', fontsize=12)
    bar = fig.colorbar(mapped, ax=axes[1], fraction=.045, pad=.02, ticks=[-90, -45, 0, 45, 90])
    bar.set_label('Angle to X axis (°)', fontsize=10)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor='white')
    plt.close(fig)


def save_noise_comparison(raw, result, noise, scale, edge, output):
    """Show both inputs, their angle maps and the measured histogram overlay."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(13.2, 7.3), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1, 1.1, 1.4])
    input_axes = [fig.add_subplot(grid[row, 0]) for row in (0, 1)]
    angle_axes = [fig.add_subplot(grid[row, 1]) for row in (0, 1)]
    histogram = fig.add_subplot(grid[:, 2])
    colors = ['#258172', '#8865ad']
    examples = [('Cell-free collagen', raw, result),
                ('Gaussian noise · example field', noise['image'], noise)]
    for row, (title, image, values) in enumerate(examples):
        cropped = image[edge:-edge, edge:-edge] if edge else image
        low, high = np.percentile(cropped, [1, 99])
        for ax in (input_axes[row], angle_axes[row]):
            ax.imshow(cropped, cmap='gray', vmin=low, vmax=high)
            ax.set_axis_off()
        mapped = angle_axes[row].imshow(values['angle_x'], cmap='gist_rainbow',
                                         vmin=-90, vmax=90, alpha=.65)
        input_axes[row].set_title(title, fontsize=12, color=colors[row])
        angle_axes[row].set_title('Angle to X axis', fontsize=12)
        if row == 0:
            for ax in (input_axes[row], angle_axes[row]):
                ax.plot([25, 25+50/scale], [cropped.shape[0]-30]*2, color='white', lw=3)
                ax.text(25, cropped.shape[0]-43, '50 µm', color='white', fontsize=9)
        label = 'Image mean' if row == 0 else f"Mean of {noise['count']} fields"
        input_axes[row].text(.5, -.04,
            f"{label}: alignment {values['mean_alignment_x']:.3f}\n"
            f"Coherency {values['mean_coherency']:.3f}",
            transform=input_axes[row].transAxes, ha='center', va='top', fontsize=10)
    colorbar = fig.colorbar(mapped, ax=angle_axes, fraction=.045, pad=.025,
                           ticks=[-90, -45, 0, 45, 90], shrink=.8)
    colorbar.set_label('Angle to X axis (°)', fontsize=10)

    bins = noise['bins']
    angles = result['angle_x']
    counts, _ = np.histogram(angles[np.isfinite(angles)], bins=bins)
    probability = counts / counts.sum()
    noise_mean = noise['histograms'].mean(axis=0)
    noise_sem = noise['histograms'].std(axis=0, ddof=1) / np.sqrt(noise['count'])
    histogram.stairs(probability, bins, color=colors[0], lw=1.8, label='Collagen · one image')
    histogram.stairs(probability, bins, color=colors[0], fill=True, alpha=.12)
    histogram.stairs(noise_mean, bins, color=colors[1], lw=1.8,
                     label=f"Noise · mean ± SEM, n = {noise['count']}")
    histogram.fill_between(bins, np.r_[noise_mean-noise_sem, (noise_mean-noise_sem)[-1]],
                           np.r_[noise_mean+noise_sem, (noise_mean+noise_sem)[-1]],
                           step='post', color=colors[1], alpha=.18)
    histogram.axhline(1 / len(probability), color='#777777', ls='--', lw=1,
                     label='Uniform angles · expectation')
    histogram.set(xlim=(-90, 90), ylim=(0, 1.22 * max(probability.max(), (noise_mean+noise_sem).max())),
                  xticks=[-90, -45, 0, 45, 90], xlabel='Angle to X axis (°)',
                  ylabel='Probability per 5° bin', title='Local tensor directions')
    histogram.spines[['top', 'right']].set_visible(False)
    histogram.legend(frameon=False, fontsize=9, loc='upper left')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', choices=['whole', 'phalloidin', 'mask'], default='whole')
    parser.add_argument('--figure', type=Path, help='Optional output PNG for the selected example')
    args = parser.parse_args()
    run(args.case, args.figure)
