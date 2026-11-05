"""Small direct-API examples using bundled images. Optionally save the collagen figure."""
from pathlib import Path
import argparse
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from CompactionAnalyzer import analyze_image


def run(data, case="all"):
    cell = None
    if case in ("all", "cell"):
        # BEGIN cell
        cell = analyze_image(
            data / "ExampleCells/Cell_1/C003.tif",  # Collagen
            data / "ExampleCells/Cell_1/C004.tif",  # Cell stain
            mode="full",
            outputs=["mean_alignment_cell", "mean_angle_cell",
                     "alignment_cell", "cell_mask"],
            scale=0.318, sigma_tensor=7 / 0.318, edge=40,
            seg_gaus1=6, seg_gaus2=80,
        )
        print(cell["mean_alignment_cell"], cell["mean_angle_cell"])
        # END cell

    if case in ("all", "whole"):
        # BEGIN whole
        collagen = next((data / "EmptyGel/Pos01").glob("*C003*.tif"))
        deviation = analyze_image(
            collagen, mode="axis_orientation_only", outputs="mean_deviation_x",
            scale=0.249, sigma_tensor=7 / 0.249, edge=40,
        )
        print(deviation)  # Degrees; use "mean_deviation_y" for Y.
        # END whole

    if case in ("all", "mask"):
        # BEGIN mask
        actin = sorted((data / "StressFibers/controlzoom1.5").glob("*.tif"))[0]
        masked = analyze_image(
            actin, mode="axis_orientation_masked",
            outputs=["mean_alignment_x", "mean_coherency"],
            scale=0.48, sigma_tensor=3 / 0.48, edge=10, mask_threshold=0.5,
        )
        print(masked)
        # For a separate mask image, add: mask_image="mask_channel.tif"
        # END mask

    if case in ("all", "point"):
        # BEGIN point
        point = analyze_image(
            data / "ExampleCells/Cell_1/C003.tif", mode="point_orientation",
            reference_point=(400, 400),
            outputs=["mean_alignment_point", "mean_angle_point"],
            scale=0.318, sigma_tensor=7 / 0.318, edge=40,
        )
        print(point)
        # END point

    if case in ("all", "line"):
        # BEGIN line
        line = analyze_image(
            data / "ExampleCells/Cell_1/C003.tif", mode="line_orientation",
            reference_line=((120, 600), (680, 200)),
            outputs=["mean_alignment_line", "mean_deviation_line", "angle_line"],
            scale=0.318, sigma_tensor=7 / 0.318, edge=40,
        )
        print(line["mean_alignment_line"], line["mean_deviation_line"])
        # line["angle_line"] is the signed angle map in degrees.
        # END line

    if case in ("all", "cell_list"):
        # BEGIN cell_list
        cell_list = sorted((data / "ExampleCells").glob("Cell_*"))
        for folder in cell_list:
            alignment = analyze_image(
                folder / "C003.tif", folder / "C004.tif", mode="full",
                outputs="mean_alignment_cell",
                scale=0.318, sigma_tensor=7 / 0.318, edge=40,
                seg_gaus1=6, seg_gaus2=80,
            )
            print(folder.name, alignment)
        # END cell_list
    return cell


def save_collagen_figure(data, cell, output):
    """Render the actual input pair, returned segmentation and alignment map."""
    import imageio.v2 as imageio
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    crop = (slice(40, -40),) * 2
    collagen = imageio.imread(data / 'ExampleCells/Cell_1/C003.tif')[crop]
    stain = imageio.imread(data / 'ExampleCells/Cell_1/C004.tif')[crop]
    mask = cell['cell_mask']
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5), constrained_layout=True)
    for ax, image, title in zip(axes, (collagen, stain, collagen),
                               ('Collagen', 'Cell stain + segmentation', 'Alignment towards cell')):
        low, high = np.percentile(image, [1, 99])
        ax.imshow(image, cmap='gray', vmin=low, vmax=high)
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_axis_off()
    # The segmentation is exactly the mask returned by analyze_image.
    overlay = np.zeros((*mask.shape, 4))
    overlay[mask] = [1, .2, .2, .28]
    axes[1].imshow(overlay)
    axes[1].contour(mask, levels=[.5], colors=['#f25f5c'], linewidths=.8)
    mapped = axes[2].imshow(cell['alignment_cell'], cmap='coolwarm', vmin=-1, vmax=1, alpha=.75)
    axes[2].contour(mask, levels=[.5], colors=['#f25f5c'], linewidths=.8)
    # 50 µm bar on the original acquisition scale, identical crop in all panels.
    for ax in axes:
        ax.plot([25, 25 + 50/.318], [collagen.shape[0]-30]*2, color='white', lw=3)
        ax.text(25, collagen.shape[0]-43, '50 µm', color='white', fontsize=9)
    bar = fig.colorbar(mapped, ax=axes[2], fraction=.045, pad=.025, ticks=[-1, 0, 1])
    bar.set_label('−1 tangential   /   +1 radial', fontsize=10)
    fig.suptitle('Cell_1 · mean alignment = ' + f'{cell["mean_alignment_cell"]:.3f}'
                 + ' · mean angular deviation = ' + f'{cell["mean_angle_cell"]:.1f}°', fontsize=12)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', type=Path, help='Optional output PNG for the collagen example')
    parser.add_argument('--case', choices=['all', 'cell', 'whole', 'mask', 'point', 'line', 'cell_list'],
                        default='all', help='Run one example, or all examples by default')
    args = parser.parse_args()
    if args.figure and args.case not in ('all', 'cell'):
        parser.error('--figure renders the cell example; use --case cell or --case all')
    data = root / 'Tutorial'
    cell = run(data, args.case)
    if args.figure:
        save_collagen_figure(data, cell, args.figure)
