"""Synthetic fibers with preferred point/line alignment and local directional variation.

python Tutorial/Example_SyntheticReferences.py --case point --output point_example
python Tutorial/Example_SyntheticReferences.py --case line --output line_example --saenopy

The optional --saenopy flag also writes portable .saenopyOrientation files.
Synthetic images illustrate the geometry; they are not biological measurements.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import imageio.v2 as imageio
from scipy.ndimage import gaussian_filter
from CompactionAnalyzer import analyze_image


def make_image(case, seed=0, bend_amplitude=5.0, noise_sigma=900.0):
    """Return an image, mask and reference in original image pixels.

    The preferred angle is 25 degrees from the radial or line reference.
    Smooth random bending adds a spread of local directions; intensity noise
    adds image grain. Set bend_amplitude=0 for the original regular geometry.
    The same seed reproduces the same image.
    """
    yy, xx = np.indices((512, 512), dtype=float)
    dx, dy = xx - 255.5, -(yy - 255.5)
    rng = np.random.default_rng(seed)
    phase_offset = rng.uniform(0, 2 * np.pi)
    # Bend the fiber pattern with smooth Gaussian random displacements.
    # Subtract the displacement at the center to keep the point defect fixed.
    # Bend amplitude and correlation length are in pixels.
    displacement = []
    for _ in range(2):
        noise = gaussian_filter(rng.normal(size=xx.shape), sigma=20)
        noise /= noise.std()
        noise -= noise[255:257, 255:257].mean()
        displacement.append(bend_amplitude * noise)
    x_bent = xx + displacement[0]
    y_bent = yy + displacement[1]
    if case == 'point':
        radius = np.hypot(x_bent - 255.5, y_bent - 255.5)
        phi = np.arctan2(-(y_bent - 255.5), x_bent - 255.5)
        phase = 32 * (phi - np.tan(np.deg2rad(25)) * np.log(np.maximum(radius, 1))) + phase_offset
        mask = (np.hypot(dx, dy) >= 65) & (np.hypot(dx, dy) <= 220)
        reference = (255.5, 255.5)
    elif case == 'line':
        theta = np.deg2rad(55)
        phase = 2 * np.pi * (x_bent * np.sin(theta) + y_bent * np.cos(theta)) / 20 + phase_offset
        mask = (dx / 205) ** 2 + (dy / 165) ** 2 <= 1
        along = np.array([np.cos(np.deg2rad(30)), -np.sin(np.deg2rad(30))]) * 175
        reference = tuple(map(tuple, np.array([255.5, 255.5]) + np.stack([-along, along])))
    else:
        raise ValueError('case must be point or line')
    signal = np.exp(-5 * np.sin(phase / 2) ** 2)
    image = np.clip(1500 + 57000 * signal + rng.normal(0, noise_sigma, signal.shape), 0, 65535).astype(np.uint16)
    return image, mask, reference

def run_case(case, output, saenopy=False):
    output=Path(output)/case
    output.mkdir(parents=True,exist_ok=True)
    bins=np.arange(0,91,1) if case=='point' else np.arange(-90,91,1)
    probabilities=[]
    summaries=[]
    first=None
    for seed in range(3):
        image,mask,reference=make_image(case,seed)
        values=analyze_image(image,mode=case+'_orientation',mask=mask,
            **{'reference_'+case:reference},sigma_tensor=5,edge=15,
            outputs=['angle_'+case,'alignment_'+case,'mean_alignment_'+case,
                     'mean_angle_point' if case=='point' else 'mean_deviation_line'])
        angle=values['angle_'+case]
        finite=angle[np.isfinite(angle)]
        counts,_=np.histogram(finite,bins=bins)
        probabilities.append(counts/counts.sum())
        deviation=values['mean_angle_point' if case=='point' else 'mean_deviation_line']
        summaries.append(dict(seed=seed,nominal_deviation_deg=25.,measured_deviation_deg=deviation,
                              angular_std_deg=float(np.std(finite)),
                              mean_alignment=values['mean_alignment_'+case]))
        image_path=output/f'image_{seed}.tif'
        mask_path=output/f'mask_{seed}.tif'
        imageio.imwrite(image_path,image)
        imageio.imwrite(mask_path,mask.astype(np.uint8)*255)
        if saenopy:
            from saenopy.gui.orientation.modules.result import ResultOrientation
            from saenopy.gui.orientation.modules.analysis import analyze_orientation
            result=ResultOrientation(output=str(output/f'{case}_{seed}.saenopyOrientation'),
                image_fiber=image_path.name,image_cell=mask_path.name,pixel_size=1.,analysis_mode=case+'_orientation')
            parameters=dict(edge=15,sigma_tensor=5.,sigma_tensor_type='pixel',sigma_first_blur=.5)
            if case=='point':
                parameters.update(reference_x=reference[0],reference_y=reference[1])
            else:
                parameters.update(zip(('line_x1','line_y1','line_x2','line_y2'),np.asarray(reference).ravel()))
            analyze_orientation(result,parameters,dict(mask_sigma=0.,mask_threshold=.5,mask_invert=False))
            np.testing.assert_allclose(result.orientation_map,values['alignment_'+case],atol=1e-12)
        if first is None:
            first=image,mask,reference,values
    probabilities=np.array(probabilities)
    np.savetxt(output/'histogram.csv',np.column_stack([bins[:-1],bins[1:],probabilities.T,probabilities.mean(axis=0)]),
               delimiter=',',header='left_deg,right_deg,image_0,image_1,image_2,mean_probability',comments='')
    (output/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    save_figure(case,*first,bins,probabilities,output/'example.png')
    print(case,': preferred angle 25 deg; measured mean',', '.join(f'{s["measured_deviation_deg"]:.2f}' for s in summaries))
    return summaries


def save_figure(case,image,mask,reference,values,bins,probabilities,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    crop=(slice(15,-15),)*2
    fig,axes=plt.subplots(1,3,figsize=(12.6,4.1),constrained_layout=True,
                          gridspec_kw=dict(width_ratios=[1,1,1.05]))
    for ax in axes[:2]:
        ax.imshow(image[crop],cmap='gray',vmin=1500,vmax=58500)
        ax.set_axis_off()
        if case=='point':
            x,y=np.asarray(reference)-15
            ax.plot(x,y,'o',color='#ffe54d',markeredgecolor='black',ms=6)
        else:
            points=np.asarray(reference)-15
            ax.axline(*points,color='#ffe54d',ls='--',lw=1.5)
            ax.plot(*points.T,'o',color='#ffe54d',markeredgecolor='black',ms=5)
        ax.set_xlim(-.5,481.5)
        ax.set_ylim(481.5,-.5)
    axes[0].contour(mask[crop],levels=[.5],colors=['#ef7e8a'],linewidths=.8)
    overlay=axes[1].imshow(values['alignment_'+case],cmap='coolwarm',vmin=-1,vmax=1,alpha=.78)
    bar=fig.colorbar(overlay,ax=axes[1],fraction=.045,pad=.025,shrink=.8,ticks=[-1,0,1])
    bar.set_label('Alignment',fontsize=10)
    axes[0].set_title('Synthetic fibers around a point' if case=='point' else 'Synthetic fibers with bending')
    axes[1].set_title('Alignment to point' if case=='point' else 'Alignment to line')
    centers=(bins[:-1]+bins[1:])/2
    mean_probability = probabilities.mean(axis=0)
    axes[2].bar(centers, mean_probability, width=np.diff(bins),
                color='#8b5db7', edgecolor='white', linewidth=.3,
                label='Mean of 3 synthetic images')
    axes[2].axvline(25,color='#333333',ls='--',lw=1,label='Preferred angle: 25°')
    axes[2].set(xlim=(0,90) if case=='point' else (-90,90),ylim=(0,1.25*mean_probability.max()),ylabel='Probability per bin',
        xlabel='Angular deviation to point (°)' if case=='point' else 'Angle to line (°)',title=f'1° bins · sum = {mean_probability.sum():.2f}')
    axes[2].set_xticks([0,25,45,90] if case=='point' else [-90,-45,0,25,45,90])
    axes[2].spines[['top','right']].set_visible(False)
    axes[2].legend(frameon=False,fontsize=8,loc='upper right' if case=='point' else 'upper left')
    fig.savefig(path,dpi=180)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',choices=['point','line','both'],default='both')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--saenopy',action='store_true',help='Also save portable results (requires Saenopy).')
    args=parser.parse_args()
    for case in ('point','line') if args.case=='both' else (args.case,):
        run_case(case,args.output,args.saenopy)
