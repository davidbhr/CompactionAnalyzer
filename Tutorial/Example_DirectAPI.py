"""Return numbers/maps from the bundled examples without writing analysis files."""
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
from CompactionAnalyzer import analyze_image

phalloidin = sorted((root / 'Tutorial/StressFibers/controlzoom1.5').glob('*.tif'))[0]
parameters = dict(scale=.48, sigma_tensor=3/.48, edge=10)

whole_image = analyze_image(phalloidin, mode='axis_orientation_only',
    outputs='mean_deviation_x', **parameters)
print('Whole-image angular deviation to X (degrees):', whole_image)

masked = analyze_image(phalloidin, mode='axis_orientation_masked',
    outputs=['mean_alignment_x', 'mean_coherency', 'angle_x'], mask_threshold=.5, **parameters)
print('Same-image mask:', {key: value.shape if hasattr(value, 'shape') else value
                           for key, value in masked.items()})

cell_folder = root / 'Tutorial/ExampleCells/Cell_1'
cell = analyze_image(cell_folder / 'C003.tif', cell_folder / 'C004.tif', mode='full',
    outputs=['mean_alignment_cell', 'mean_angle_cell'],
    scale=.318, sigma_tensor=7/.318, edge=40, seg_gaus1=6, seg_gaus2=80)
print('Cell-relative analysis:', cell)

# A chosen center or a fixed direction, using the same tensor calculation.
point = analyze_image(phalloidin, mode='point_orientation', reference_point=(512, 512),
    outputs=['mean_alignment_point', 'mean_angle_point'], **parameters)
print('Point-relative analysis:', point)

line = analyze_image(phalloidin, mode='line_orientation', reference_line=((160, 700), (800, 300)),
    outputs=['mean_alignment_line', 'mean_deviation_line'], **parameters)
print('Line-relative analysis:', line)
