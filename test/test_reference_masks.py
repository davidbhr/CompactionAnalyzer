import numpy as np
import pytest
from CompactionAnalyzer import analyze_image
from CompactionAnalyzer.CompactionFunctions import create_axis_mask

@pytest.mark.parametrize('mode',['point','line'])
@pytest.mark.parametrize('source',['boolean','same','second'])
@pytest.mark.parametrize('edge',[0,10])
def test_reference_mask_selects_original_tensor_without_moving_reference(mode,source,edge):
    yy,xx=np.indices((100,140))
    image=2+np.cos(.3*xx+.1*yy)
    mask_image=((xx-70)**2+(yy-50)**2<35**2).astype(float)
    reference={'reference_point':(70,50)} if mode=='point' else {'reference_line':((10,80),(120,20))}
    options=dict(mode=mode+'_orientation',sigma_tensor=4,edge=edge,**reference)
    outputs=['angle_x','coherency','alignment_'+mode,'mean_alignment_'+mode,'selection_mask']
    raw=analyze_image(image,outputs=outputs,**options)
    crop=(slice(edge,-edge),)*2 if edge else (slice(None),)*2
    if source=='boolean':
        mask=mask_image.astype(bool)
        extra=dict(mask=mask)
        expected=mask[crop]
    else:
        channel=image if source=='same' else mask_image
        extra=dict(mask_image=channel,mask_sigma=2.,mask_threshold=.6)
        expected=create_axis_mask(channel[crop],2.,.6,False)
    masked=analyze_image(image,outputs=outputs,**extra,**options)
    np.testing.assert_array_equal(masked['selection_mask'],expected)
    for name in ['angle_x','coherency','alignment_'+mode]:
        np.testing.assert_allclose(masked[name],np.where(expected,raw[name],np.nan),equal_nan=True)
    assert masked['mean_alignment_'+mode]==pytest.approx(np.nanmean(raw['alignment_'+mode][expected]))
    empty=analyze_image(image,mask=np.zeros_like(image,dtype=bool),outputs='mean_alignment_'+mode,**options)
    assert np.isnan(empty)

def test_point_mask_intersects_distance_selection():
    yy,xx=np.indices((100,140))
    mask=xx>70
    selected=analyze_image(np.cos(xx*.3),mode='point_orientation',reference_point=(70,50),
        mask=mask,outputs='selection_mask',min_dist=10,max_dist=30,edge=10)
    distance=np.hypot(xx-70,yy-50)
    np.testing.assert_array_equal(selected,(mask&(distance>=10)&(distance<=30))[10:-10,10:-10])
