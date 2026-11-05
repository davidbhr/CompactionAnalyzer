"""Portable Saenopy ResultOrientation 1.1 export, without importing Qt.

This adapter writes the public NPZ container used by saenopy.saveable.Saveable.
Analysis stays in CompactionAnalyzer; its full workflow supplies the complete
cell-relative statistics when a desktop project is requested.
"""
import io
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import tifffile
from CompactionAnalyzer.CompactionFunctions import StuctureAnalysisMain
from CompactionAnalyzer.plotting import line_reference_angle
from CompactionAnalyzer.distance import point_distance_profile


def flatten(data):
    """Encode nested values using Saveable's list/tuple/dict marker convention."""
    result = {}
    def visit(value, key):
        if value is None:
            result[key] = '__NONE__'
        elif isinstance(value, (list, tuple, dict)):
            result[key] = type(value).__name__
            entries = value.items() if isinstance(value, dict) else enumerate(value)
            for child_key, child in entries:
                visit(child, f'{key}/{child_key}')
        else:
            result[key] = value
    for key, value in data.items():
        visit(value, key)
    return result


def export_project(last, images, scale):
    if scale is None or not np.isfinite(scale) or scale <= 0:
        raise ValueError('A measured positive pixel size is required for Saenopy export.')
    values, metadata = last['arrays'], last['metadata']
    mode = metadata['analysis_mode']
    p = metadata['parameters'].copy()
    image = images['orientation']['array']
    source = metadata['mask_source']
    second = image if source == 'same' else images['mask']['array'] if source in ('second','cell_image') else None
    op = dict(sigma_tensor=p['sigma_tensor'], sigma_tensor_type='pixel', edge=p['edge'],
              sigma_first_blur=p['sigma_first_blur'], max_dist=None, ignore_cell_outline=False,
              angle_sections=5, shell_width=5., shell_width_type='pixel', SaveNumpy=False, plotting=False)
    sp = dict(thresh=p.get('segmention_thres', 1.), gauss1=p.get('seg_gaus1',6.),
              gauss2=p.get('seg_gaus2',80.), invert=p.get('seg_invert',False),
              mask_sigma=p.get('mask_sigma',5.), mask_threshold=p.get('mask_threshold',.5),
              mask_invert=p.get('mask_invert',False))
    data = dict(analysis_mode=mode, validity_mask=values['selection_mask'],
                angle_to_x_map=values['angle_x'], angle_to_y_map=values['angle_y'],
                coherence_map=values['coherency'], orientation_vectors=None,
                image_fiber='data/orientation.tif',
                image_cell='data/orientation.tif' if source=='same' else 'data/mask.tif' if second is not None else None,
                output='measurement.saenopyOrientation', pixel_size=float(scale),
                orientation_parameters=op, segmentation_parameters=sp,
                segmentation=None, shape=tuple(image.shape), results_distance=[],
                ___save_name__='ResultOrientation', ___save_version__='1.1')
    valid=values['selection_mask'] & np.isfinite(values['angle_x']) & np.isfinite(values['angle_y'])
    summary={'Selected pixels':int(values['selection_mask'].sum()), 'Valid pixels':int(valid.sum())}
    bins=np.linspace(-90,90,37)
    histograms=[]
    for axis in ('x','y'):
        summary['Alignment '+axis]=metadata['summary']['mean_alignment_'+axis]
        summary['Mean absolute deviation '+axis]=metadata['summary']['mean_deviation_'+axis]
        histograms.append(np.histogram(values['angle_'+axis][valid],bins=bins)[0])
    histogram=np.column_stack([bins[:-1],bins[1:],*histograms])
    summary['Mean Coherency'] = float(last['metadata']['summary']['mean_coherency'])
    data['axis_histogram'] = histogram
    if mode == 'full':
        # The desktop's weighted statistics and distance shells are computed by
        # the same existing full pipeline, not approximated from API means.
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            tifffile.imwrite(tmp/'orientation.tif', image)
            tifffile.imwrite(tmp/'cell.tif', second)
            result = StuctureAnalysisMain(fiber_list=[str(tmp/'orientation.tif')],
                cell_list=[str(tmp/'cell.tif')], out_list=[str(tmp/'results')],
                scale=float(scale), mode_saenopy=True, return_segmentation=True,
                shell_width=5., angle_sections=5, SaveNumpy=False, SaveExcel=False,
                plotting=False, **p)
        angle, orientation, coherency, vectors, total, angles, distance, segmentation = result
        np.testing.assert_allclose(orientation[values['selection_mask']],
                                   values['alignment_cell'][values['selection_mask']], atol=1e-10, equal_nan=True)
        data.update(orientation_map=orientation, coherence_map=coherency,
                    orientation_vectors=vectors, segmentation=segmentation,
                    results_distance=distance.to_dict(orient='records'))
        cell_summary=total.to_dict(orient='records')[0]
        # Keep the full-workflow coherency definition in the full desktop file.
        summary.pop('Mean Coherency', None)
        cell_summary.update(summary)
        summary=cell_summary
    elif mode in ('point_orientation', 'line_orientation'):
        suffix='point' if mode=='point_orientation' else 'line'
        data['orientation_map']=values['alignment_'+suffix]
        summary['Alignment '+suffix]=metadata['summary']['mean_alignment_'+suffix]
        summary['Mean angle point' if suffix=='point' else 'Mean deviation line']=metadata['summary']['mean_deviation_'+suffix]
        if suffix=='line':
            points=np.asarray(p['reference_line'])
            op.update(zip(('line_x1','line_y1','line_x2','line_y2'), points.ravel().tolist()))
            summary.update(zip(('Reference line x1 (px)','Reference line y1 (px)',
                                'Reference line x2 (px)','Reference line y2 (px)'), points.ravel().tolist()))
            summary['Reference line angle (deg)']=line_reference_angle(points)
        else:
            x,y=p['reference_point']
            op.update(reference_x=float(x),reference_y=float(y))
            summary.update({'Reference x (px)':float(x), 'Reference y (px)':float(y)})
            profile=point_distance_profile(values['alignment_point'],values['coherency'],
                (x-p['edge'],y-p['edge']), mask=values['selection_mask'], shell_width=5.,scale=float(scale))
            columns={'distance_start_px':'Shell_start (px)', 'distance_end_px':'Shell_end (px)',
                'distance_px':'Shell_mid (px)', 'distance_um':'Shell_mid (µm)',
                'mean_alignment_point':'Point alignment (shell)', 'mean_angle_point':'Point angular deviation (shell)',
                'mean_coherency':'Point coherency (shell)', 'valid_pixels':'Valid pixels', 'coherency_pixels':'Coherency pixels'}
            data['results_distance']=[{column:profile[key][i].item() for key,column in columns.items()}
                                      for i in range(len(profile['distance_px']))]
    else:
        data['orientation_map']=values['alignment_x']
    data['results_total']=[summary]
    flattened=flatten(data)
    if any(np.asarray(value).dtype.hasobject for value in flattened.values()):
        raise ValueError('Saenopy exports must not contain object arrays.')
    measurement=io.BytesIO()
    np.savez_compressed(measurement, **flattened)
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as output:
        output.writestr('measurement.saenopyOrientation',measurement.getvalue())
        for name,array in [('orientation',image),('mask',second if source!='same' else None)]:
            if array is None:
                continue
            stream=io.BytesIO()
            tifffile.imwrite(stream,array)
            output.writestr('data/'+name+'.tif',stream.getvalue())
        output.writestr('OPEN_IN_SAENOPY.txt',
            'Extract this ZIP completely. Keep the data folder next to the measurement file.\n'
            'Drag measurement.saenopyOrientation into the Orientation GUI.\n'
            'Use the updated Saenopy version supporting ResultOrientation 1.1 and all reference modes.\n'
            'The images are the full-resolution grayscale inputs used for this analysis.\n'
            'Full cell-relative desktop exports also contain the original weighted statistics and distance shells.\n')
    return archive.getvalue()
