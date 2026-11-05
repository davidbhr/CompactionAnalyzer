"""Three biological examples using the direct Python API.

python Tutorial/Example_BiologicalOrientation.py --case cardiomyocyte --figure cell.png
python Tutorial/Example_BiologicalOrientation.py --case spheroid --figure spheroid.png
python Tutorial/Example_BiologicalOrientation.py --case lamellipodia --figure lamellipodia.png

Add --saenopy OUTPUT_FOLDER to save a portable GUI result. The lamellipodia
image has no known calibration; that export additionally needs --pixel-size
with the measured micrometers per pixel. Its API analysis works in pixels.
"""
from pathlib import Path
import argparse
import sys
import numpy as np
import imageio.v2 as imageio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from CompactionAnalyzer import analyze_image

DATA = Path(__file__).resolve().parent/'BiologicalExamples'


def run(case, figure=None, saenopy=None, pixel_size=None, data=DATA,
        distance_figure=None, shell_width=10.):
    data = Path(data)
    if case == 'cardiomyocyte':
        source = data/'cardiomyocyte_05/phalloidin.tif'
        mask_image = data/'cardiomyocyte_05/actinin.tif'
        scale, sigma, edge = 1/6.411017, 8., 15
        mode = 'axis_orientation_masked'
        segmentation = dict(mask_sigma=8., mask_threshold=.65, mask_invert=False)
        parameters = dict(mask_image=mask_image, **segmentation)
        map_name = 'angle_x'
        summaries = ['mean_deviation_x', 'mean_alignment_x', 'mean_coherency']
        point = None
    elif case in ('spheroid', 'lamellipodia'):
        source = data/case/'orientation.tif'
        mask_image = data/case/'selection.tif'
        # Original, uncropped (x, y) pixel coordinates: x right, y down.
        point = (1200, 910) if case == 'spheroid' else (390, 325)
        scale = 1. if case == 'spheroid' else None
        sigma, edge = (6., 15) if case == 'spheroid' else (3., 15)
        mode = 'point_orientation'
        segmentation = dict(mask_sigma=0., mask_threshold=.5, mask_invert=False)
        parameters = dict(mask=imageio.imread(mask_image)>0, reference_point=point)
        map_name = 'alignment_point'
        summaries = ['mean_angle_point', 'mean_alignment_point', 'mean_coherency']
    else:
        raise ValueError('Choose cardiomyocyte, spheroid or lamellipodia.')

    if pixel_size is not None:
        if not np.isfinite(pixel_size) or pixel_size <= 0:
            raise ValueError('pixel_size must be positive and finite.')
        scale = pixel_size
    if saenopy is not None and scale is None:
        raise ValueError('The lamellipodia image has no known pixel size. Supply a measured --pixel-size for GUI export; omit --saenopy for the uncalibrated API example.')

    extra_outputs = []
    if distance_figure is not None:
        if mode != 'point_orientation':
            raise ValueError('Distance shells in this example require spheroid or lamellipodia.')
        parameters['shell_width'] = shell_width
        extra_outputs = ['distance_profile']
    values = analyze_image(
        source, mode=mode, **parameters, scale=scale or 1., sigma_tensor=sigma, edge=edge,
        outputs=[map_name, 'coherency', 'selection_mask', *summaries, *extra_outputs],
    )
    print(case, {name: values[name] for name in summaries})
    if figure is not None:
        save_figure(source, mask_image, values, case, point, scale, edge, figure)
    if distance_figure is not None:
        save_distance_figure(source, values, case, point, scale, edge, distance_figure)
    if saenopy is not None:
        save_result(source, mask_image, values, case, mode, point, scale, sigma, edge, segmentation, saenopy,
                    shell_width=shell_width if distance_figure is not None else None)
    return values


def save_distance_figure(source, values, case, point, scale, edge, output):
    """One image's local means, without treating neighboring pixels as replicates."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    image = imageio.imread(source)
    profile = values['distance_profile']
    distance = profile['distance_px'] if scale is None else profile['distance_um']
    unit = 'px' if scale is None else 'µm'
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.1), constrained_layout=True)
    axes[0].imshow(image[edge:-edge,edge:-edge],cmap='gray',vmin=0,vmax=np.percentile(image,99.5))
    axes[0].contour(values['selection_mask'], levels=[.5], colors=['#ef7e8a'], linewidths=.7)
    center = np.asarray(point)-edge
    axes[0].plot(*center, '+', color='#ffe54d', ms=12, mew=2)
    radii = [80,120,160,200] if case == 'lamellipodia' else [300,400,500,600]
    for radius in radii:
        axes[0].add_patch(Circle(center,radius,fill=False,color='#ffe54d',ls='--',lw=.6,alpha=.7))
    limits = (160,620,555,95) if case == 'lamellipodia' else (500,1900,1610,210)
    axes[0].set(xlim=(limits[0]-edge,limits[1]-edge),ylim=(limits[2]-edge,limits[3]-edge),
                title='Point, selected area and distance rings')
    axes[0].axis('off')
    for ax, key, title, limits in [
        (axes[1], 'mean_alignment_point', 'Alignment towards point', (-1,1)),
        (axes[2], 'mean_coherency', 'Local coherency', (0,1))]:
        ax.plot(distance,profile[key],'-o',color='#8b5db7',ms=3)
        ax.set(xlabel=f'Distance from reference point ({unit})',ylabel='Mean within ring',
               title=title,ylim=limits)
        ax.spines[['top','right']].set_visible(False)
    axes[1].axhline(0,color='gray',lw=.7,ls='--')
    width = profile['distance_end_px'][0]-profile['distance_start_px'][0]
    fig.suptitle(f'One image · ring width {width*(scale or 1.):g} {unit} · selected pixels only',fontsize=11)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=170,facecolor='white');plt.close(fig)


def save_figure(source, mask_image, values, case, point, scale, edge, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    image = imageio.imread(source)
    crop = (slice(edge, -edge),)*2
    gray = image[crop]
    cardiomyocyte = case == 'cardiomyocyte'
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.5), constrained_layout=True)
    for ax in axes:
        ax.imshow(gray, cmap='gray', vmin=0, vmax=np.percentile(gray, 99.5))
        ax.axis('off')
    if cardiomyocyte:
        actinin = imageio.imread(mask_image)[crop]
        axes[1].images[0].set_data(actinin)
        axes[1].images[0].set_clim(0, np.percentile(actinin,99.5))
        overlay = np.zeros((*gray.shape,4))
        overlay[values['selection_mask']] = [1., .2, .2, .3]
        axes[1].imshow(overlay)
        mapped = axes[2].imshow(values['angle_x'], cmap='gist_rainbow', vmin=-90, vmax=90, alpha=.75)
        axes[0].set_title('Phalloidin · actin fibers')
        axes[1].set_title('α-Actinin + selection mask')
        axes[2].set_title('Angle to X axis within mask')
        bar = fig.colorbar(mapped, ax=axes[2], fraction=.045, pad=.02, ticks=[-90,-45,0,45,90])
        bar.set_label('Angle (°)')
    else:
        # Show the selected cell structures at a useful magnification. Reference
        # coordinates still refer to the complete original image.
        limits = (550,1850,1510,210) if case == 'spheroid' else (210,570,480,120)
        for ax in axes[:2]:
            ax.plot(point[0]-edge,point[1]-edge,'+',color='#ffe54d',ms=13,mew=2)
            ax.set_xlim(limits[0]-edge,limits[1]-edge)
            ax.set_ylim(limits[2]-edge,limits[3]-edge)
        axes[0].contour(values['selection_mask'],levels=[.5],colors=['#ef7e8a'],linewidths=.5)
        mapped = axes[1].imshow(values['alignment_point'],cmap='coolwarm',vmin=-1,vmax=1,alpha=.8)
        axes[0].set_title('Spheroid · center selected' if case=='spheroid' else 'Lamellipodia · cell point selected')
        axes[1].set_title('Alignment to selected point')
        bar=fig.colorbar(mapped,ax=axes[1],fraction=.045,pad=.02,ticks=[-1,0,1])
        bar.set_label('Alignment')
        axes[2].clear()
        axes[2].set_axis_on()
        axes[2].set_aspect('auto')
        finite = values['alignment_point'][np.isfinite(values['alignment_point'])]
        angles = (1-finite)*45
        counts,bins=np.histogram(angles,bins=np.arange(0,91,2))
        probability=counts/counts.sum()
        axes[2].bar((bins[:-1]+bins[1:])/2,probability,width=2,color='#8b5db7',edgecolor='white',lw=.3)
        axes[2].set(xlim=(0,90),xlabel='Angular deviation to point (°)',ylabel='Probability per bin',
                    title=f'2° bins · sum = {probability.sum():.2f}')
        axes[2].spines[['top','right']].set_visible(False)
    for ax in axes if cardiomyocyte else axes[:2]:
        left,right=ax.get_xlim();bottom,top=ax.get_ylim()
        length = 20/scale if cardiomyocyte else (200/scale if case=='spheroid' else 50)
        label = '20 µm' if cardiomyocyte else ('~200 µm' if case=='spheroid' else (f'{50*scale:g} µm' if scale else '50 px'))
        x,y=left+.06*(right-left),bottom-.08*(bottom-top)
        ax.plot([x,x+length],[y,y],color='white',lw=3)
        ax.text(x,y-.025*(bottom-top),label,color='white',fontsize=9)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=170,facecolor='white');plt.close(fig)


def save_result(source, mask_image, values, case, mode, point, scale, sigma, edge, segmentation, output, reference_line=None, shell_width=None):
    """Store images alongside the result so the whole folder can be moved."""
    import shutil
    from saenopy.gui.orientation.modules.result import ResultOrientation
    from saenopy.gui.orientation.modules.analysis import analyze_orientation
    output=Path(output)/case;output.mkdir(parents=True,exist_ok=True)
    for src,name in [(source,'orientation.tif'),(mask_image,'mask.tif')]:
        if src is not None:
            shutil.copyfile(src,output/name)
    result=ResultOrientation(output=str(output/f'{case}.saenopyOrientation'),
        image_fiber='orientation.tif',image_cell='mask.tif' if mask_image is not None else None,
        pixel_size=scale,analysis_mode=mode)
    result.shape=imageio.imread(source).shape[:2]
    params=dict(edge=edge,sigma_tensor=sigma,sigma_tensor_type='pixel',sigma_first_blur=.5,SaveNumpy=False,plotting=False)
    if shell_width is not None:
        params.update(shell_width=shell_width,shell_width_type='pixel')
    if point is not None:
        params.update(reference_x=point[0],reference_y=point[1])
    if reference_line is not None:
        params.update(zip(('line_x1','line_y1','line_x2','line_y2'), np.asarray(reference_line).ravel()))
    analyze_orientation(result,params,segmentation)
    reference_mode = mode in ('point_orientation','line_orientation')
    key = 'alignment_' + mode.split('_')[0] if reference_mode else 'angle_x'
    stored = result.orientation_map if reference_mode else result.angle_to_x_map
    np.testing.assert_allclose(np.where(result.validity_mask,stored,np.nan),values[key],atol=1e-10)
    np.testing.assert_array_equal(result.validity_mask,values['selection_mask'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=['cardiomyocyte','spheroid','lamellipodia'],required=True)
    parser.add_argument('--figure',type=Path)
    parser.add_argument('--saenopy',type=Path)
    parser.add_argument('--pixel-size',type=float,help='Measured µm/pixel; required for lamellipodia GUI export.')
    parser.add_argument('--data',type=Path,default=DATA)
    parser.add_argument('--distance-figure',type=Path,help='Point-distance alignment and coherency plot.')
    parser.add_argument('--shell-width',type=float,default=10.,help='Distance-ring width in pixels.')
    args=parser.parse_args()
    run(args.case,args.figure,args.saenopy,args.pixel_size,args.data,
        distance_figure=args.distance_figure,shell_width=args.shell_width)
