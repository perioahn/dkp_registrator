from dataclasses import replace

import cv2
import numpy as np
import pytest

from config import DEFAULT
import regional_matching as r
from sliding_windows import (paired_window,propose_windows,crop_pair,inside_window,
                             overlap,spread_indices,neighborhoods,density_weights,
                             SearchBudget,followup_window)


def points():
    return np.array([(x,y) for y in range(40,980,40) for x in range(40,980,40)],float)


@pytest.mark.parametrize('scale',[.7,1.,1.3])
def test_sliding_windows_preserve_relative_scale_and_overlap(scale):
    p0 = points()
    p1 = p0*scale+[20,30]
    windows = propose_windows(p0,p1,np.ones(len(p0)),(1024,1024),(1400,1400))
    assert len(windows) > 6
    for window in windows:
        assert window.side0 == pytest.approx(1024*.22,abs=.01)
        assert window.side1/window.side0 == pytest.approx(scale,abs=1e-5)
        np.testing.assert_allclose(window.center1,window.center0*scale+[20,30],atol=.01)
    assert any(.1 < overlap(a,b) < .7 for i,a in enumerate(windows) for b in windows[i+1:])


def test_windows_do_not_depend_on_an_artificial_quarter_boundary():
    # A compact object straddles x=256,y=256; none of its four quadrants alone
    # has enough support for the previous fixed-cell seed rule.
    p = np.array([[244,244],[251,269],[270,250],[269,269],[248,255],[260,247]],float)
    windows = propose_windows(p,p+[11,4],np.ones(len(p)),(1024,1024),(1024,1024))
    assert windows
    assert any(inside_window(p,p+[11,4],w).all() for w in windows)


def test_sampling_and_neighborhoods_translate_with_data():
    p=points()
    shifted=p+[13.25,-17.5]
    np.testing.assert_array_equal(spread_indices(p,np.ones(len(p)),24),
                                  spread_indices(shifted,np.ones(len(p)),24))
    np.testing.assert_array_equal(neighborhoods(p,(1024,1024)),neighborhoods(shifted,(1024,1024)))
    np.testing.assert_allclose(density_weights(p,p,(1024,1024),(1024,1024)),
                               density_weights(shifted,shifted,(1024,1024),(1024,1024)))


@pytest.mark.parametrize('center',[[0,0],[1023,0],[0,1023],[1023,1023]])
def test_boundary_shift_preserves_window_size_and_paired_centers(center):
    M=np.array([[1.2,0,30],[0,1.2,20]],float)
    window=paired_window(center,220,M,(1024,1024),(1300,1300))
    assert window.side0 == 220 and window.side1 == 264
    assert (window.center0>=110).all() and (window.center0<=914).all()
    np.testing.assert_allclose(window.center1,window.center0*1.2+[30,20])


def test_scale_normalization_samples_same_content_without_axis_stretch():
    rng=np.random.default_rng(15)
    image=cv2.GaussianBlur(rng.integers(0,255,(512,512),dtype=np.uint8),(9,9),0)
    # Generate exactly known coordinates instead of relying on resize half-pixel conventions.
    moving=cv2.warpAffine(image,np.array([[1.3,0,10],[0,1.3,20]],float),(700,700))
    window=paired_window([260,240],220,np.array([[1.3,0,10],[0,1.3,20]],float),image.shape,moving.shape)
    f,m,(fs,fo),(ms,mo)=crop_pair(image,moving,window,1,1,640,2)
    assert f.shape == m.shape and max(f.shape)<=640
    p=np.array([[230,220],[280,270]],float)
    np.testing.assert_allclose((p-fo)*fs,(p*1.3+[10,20]-mo)*ms,atol=1e-7)
    assert np.mean(np.abs(f[10:-10].astype(float)-m[10:-10])) < 8


def test_padding_correspondences_are_rejected_in_original_frame():
    M=np.array([[1,0,-300],[0,1,0]],float)
    window=paired_window([10,10],220,M,(200,200),(200,200))
    _,_,(fs,fo),(ms,mo)=crop_pair(np.zeros((200,200),np.uint8),np.zeros((200,200),np.uint8),
                                window,1,1,320,2)
    k=np.array([[0,0],[100,100]],float)
    p0,p1=k/fs+fo,k/ms+mo
    kept=r.bounded_pool(p0,p1,np.ones(2),(200,200),(200,200))
    assert not len(kept[0])


def test_dense_cluster_does_not_suppress_sparse_distant_points():
    rng=np.random.default_rng(0)
    p=np.r_[rng.uniform(10,180,(1800,2)),[[800,800],[920,900],[950,800]]]
    cf=np.r_[np.full(1800,.99),[.6,.6,.6]]
    out,_,_=r.bounded_pool(p,p,cf,(1024,1024),(1024,1024),limit=64)
    assert np.sum(out[:,0]>700) == 3


def test_followup_is_finite_and_follows_edge_support():
    M=np.array([[1,0,10],[0,1,20]],float)
    w=paired_window([400,400],220,M,(1024,1024),(1024,1024))
    p=np.array([(x,y) for x in range(470,500,5) for y in range(380,430,5)],float)
    nxt=followup_window(w,p,p+[10,20],(1024,1024),(1024,1024))
    assert nxt.depth == 1 and nxt.center0[0]>w.center0[0] and nxt.side0 == w.side0
    assert followup_window(nxt,p,p+[10,20],(1024,1024),(1024,1024)) is None


def test_shared_budget_limits_multiple_refinement_calls():
    p=points()*.6
    image=np.zeros((640,640,3),np.uint8)
    base={'status':'warn','metrics':{},'M_full':np.eye(3),'registered_img':image}
    budget=SearchBudget()
    calls=[]
    def empty(*a,**kw):
        calls.append(1)
        return np.empty((0,2)),np.empty((0,2)),np.empty(0)
    cfg=replace(DEFAULT,regional_max_crops=2,regional_budget_seconds=60)
    for _ in range(3):
        r.refine_result(image,image,base,[(p,p,np.ones(len(p)))],cfg,match_fn=empty,budget=budget)
    assert len(calls)==2 and budget.calls==2


def test_lazy_shares_one_budget_across_orientations(monkeypatch):
    import register
    budgets=[]
    monkeypatch.setattr(register,'_prescreen_orientations',lambda *a,**kw:
                        [((0,0),False,k,str(k)) for k in range(4)])
    def fake(*args,**kw):
        budgets.append(kw['_regional_budget'])
        return [{'status':'warn','metrics':{'n_inlier':20},'label':'test'}]
    monkeypatch.setattr(register,'register_test',fake)
    im=np.zeros((40,40,3),np.uint8); mask=np.full((40,40),255,np.uint8)
    register.register_test_lazy(im,im,mask,mask)
    assert len(budgets)==4 and all(b is budgets[0] for b in budgets)
