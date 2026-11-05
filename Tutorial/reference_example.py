"""Shared runner for the point and line examples; calculations use analyze_image."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from CompactionAnalyzer import analyze_image


def run_reference(mode, image=None, *, point=(400, 400), line=((120, 600), (680, 200)),
                  scale=.318, sigma_um=7, edge=40, figure=None):
    image = Path(image) if image is not None else ROOT/'Tutorial/ExampleCells/Cell_1/C003.tif'
    reference = {'reference_point': point} if mode == 'point' else {'reference_line': line}
    deviation = 'mean_angle_point' if mode == 'point' else 'mean_deviation_line'
    result = analyze_image(image, mode=mode+'_orientation', **reference,
        outputs=[f'mean_alignment_{mode}', deviation, f'alignment_{mode}', f'angle_{mode}'],
        scale=scale, sigma_tensor=sigma_um/scale, edge=edge)
    print(f'{mode}: alignment={result[f"mean_alignment_{mode}"]:.6f}; '
          f'angular deviation={result[deviation]:.3f} deg')
    if figure is not None:
        save_figure(image, result, mode, point, line, edge, scale, figure)
    return result


def save_figure(image_path, result, mode, point, line, edge, scale, output):
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    source = imageio.imread(image_path)
    cropped = source[edge:-edge, edge:-edge] if edge else source
    low, high = np.percentile(cropped, [1, 99])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.4), constrained_layout=True)
    for ax in axes:
        ax.imshow(cropped, cmap='gray', vmin=low, vmax=high)
        ax.set_axis_off()
    mapped = axes[1].imshow(result[f'alignment_{mode}'], cmap='coolwarm', vmin=-1, vmax=1, alpha=.72)
    for ax in axes:
        if mode == 'point':
            x, y = np.asarray(point)-edge
            ax.plot(x, y, 'o', color='#ffe54d', markeredgecolor='black', markersize=7)
        else:
            points = np.asarray(line)-edge
            ax.axline(points[0], points[1], color='#ffe54d', linestyle='--', linewidth=1.2)
            ax.plot(points[:, 0], points[:, 1], '-o', color='#ffe54d', markeredgecolor='black', markersize=5)
        ax.set_xlim(-.5, cropped.shape[1]-.5)
        ax.set_ylim(cropped.shape[0]-.5, -.5)
        ax.plot([25, 25+50/scale], [cropped.shape[0]-30]*2, color='white', lw=3)
        ax.text(25, cropped.shape[0]-43, '50 µm', color='white', fontsize=9)
    axes[0].set_title('Collagen + selected ' + mode, fontsize=12)
    axes[1].set_title('Alignment ' + ('towards point' if mode == 'point' else 'relative to line'), fontsize=12)
    colorbar = fig.colorbar(mapped, ax=axes[1], fraction=.045, pad=.02, ticks=[-1, 0, 1])
    colorbar.set_label('−1 tangential / +1 radial' if mode == 'point' else '−1 perpendicular / +1 parallel', fontsize=10)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor='white')
    plt.close(fig)


def main(mode):
    parser = argparse.ArgumentParser(description=f'Orientation relative to a {mode} in a bundled collagen image.')
    parser.add_argument('--image', type=Path, default=None, help='Optional replacement orientation image')
    parser.add_argument('--scale', type=float, default=.318, help='Pixel size in µm; bundled image: 0.318')
    parser.add_argument('--sigma-um', type=float, default=7., help='Tensor Gaussian sigma in µm')
    parser.add_argument('--edge', type=int, default=40, help='Pixels cropped from each edge')
    parser.add_argument('--figure', type=Path, help='Optional output PNG')
    if mode == 'point':
        parser.add_argument('--point', nargs=2, type=float, default=(400, 400), metavar=('X', 'Y'))
    else:
        parser.add_argument('--line', nargs=4, type=float, default=(120, 600, 680, 200), metavar=('X1', 'Y1', 'X2', 'Y2'))
    args = parser.parse_args()
    reference = {'point': args.point} if mode == 'point' else {'line': (args.line[:2], args.line[2:])}
    run_reference(mode, args.image, **reference, scale=args.scale, sigma_um=args.sigma_um,
                  edge=args.edge, figure=args.figure)
