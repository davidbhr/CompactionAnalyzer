import numpy as np
import pytest
from CompactionAnalyzer import analyze_image
from CompactionAnalyzer.distance import point_distance_profile


def test_ring_boundaries_gaps_counts_and_scale():
    alignment=np.array([[1., np.nan, 0., -1., 1.]])
    coherency=np.array([[.2, .4, np.nan, .8, 1.]])
    mask=np.array([[True, True, False, True, True]])
    p=point_distance_profile(alignment,coherency,(0,0),mask=mask,shell_width=1,scale=.25)
    np.testing.assert_allclose(p['distance_px'],[.5,1.5,2.5,3.5,4.5])
    np.testing.assert_allclose(p['distance_um'],p['distance_px']*.25)
    np.testing.assert_allclose(p['mean_alignment_point'],[1,np.nan,np.nan,-1,1],equal_nan=True)
    np.testing.assert_allclose(p['mean_angle_point'],[0,np.nan,np.nan,90,0],equal_nan=True)
    np.testing.assert_array_equal(p['valid_pixels'],[1,0,0,1,1])
    np.testing.assert_array_equal(p['coherency_pixels'],[1,1,0,1,1])
    empty=point_distance_profile(alignment,coherency,(0,0),mask=np.zeros_like(mask))
    assert all(len(values)==0 for values in empty.values())


@pytest.mark.parametrize('width',[0,-1,np.nan,np.inf,1e-300,True])
def test_invalid_width(width):
    with pytest.raises(ValueError):
        point_distance_profile(np.ones((8,8)),np.ones((8,8)),(2,2),shell_width=width)


@pytest.mark.parametrize('point',[(40.25,32.5),(2.,3.)])
def test_api_profile_matches_returned_maps_and_crop(point,monkeypatch,tmp_path):
    import CompactionAnalyzer.api as api
    yy,xx=np.indices((75,91))
    image=100+90*np.cos(xx*.4+yy*.15)
    mask=(xx>14)&(xx<80)&(yy>12)
    original_image,original_mask=image.copy(),mask.copy()
    calls=[]
    original=api._cf.analyze_local
    def counted(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(api._cf,'analyze_local',counted)
    monkeypatch.chdir(tmp_path)
    result=analyze_image(image,mode='point_orientation',reference_point=point,mask=mask,
        sigma_tensor=2,edge=5,shell_width=7.5,scale=.3,min_dist=10,max_dist=50,
        outputs=['alignment_point','angle_point','coherency','distance_profile'])
    assert len(calls)==1
    assert not list(tmp_path.iterdir())
    np.testing.assert_array_equal(image,original_image)
    np.testing.assert_array_equal(mask,original_mask)
    radius=np.hypot(xx-point[0],yy-point[1])[5:-5,5:-5]
    p=result['distance_profile']
    for i,(start,end) in enumerate(zip(p['distance_start_px'],p['distance_end_px'])):
        region=(radius>=start)&(radius<end)
        for field,key,countkey in [('alignment_point','mean_alignment_point','valid_pixels'),
                                   ('angle_point','mean_angle_point','valid_pixels'),
                                   ('coherency','mean_coherency','coherency_pixels')]:
            valid=region&np.isfinite(result[field])
            assert p[countkey][i]==valid.sum()
            expected=result[field][valid].mean() if valid.any() else np.nan
            np.testing.assert_allclose(p[key][i],expected,atol=1e-12,equal_nan=True)


def test_only_point_mode_accepts_profile():
    image=np.ones((30,30))
    for kwargs in [dict(outputs='distance_profile'),dict(shell_width=5)]:
        with pytest.raises(ValueError,match='point_orientation'):
            analyze_image(image,edge=2,sigma_tensor=2,**kwargs)


def test_flat_and_empty_point_selection():
    image=np.ones((30,30))
    for mask in [np.ones_like(image,dtype=bool),np.zeros_like(image,dtype=bool)]:
        p=analyze_image(image,mode='point_orientation',reference_point=(15,15),mask=mask,
            outputs='distance_profile',edge=2,sigma_tensor=2)
        assert not p['valid_pixels'].any()
        assert np.isnan(p['mean_alignment_point']).all()
