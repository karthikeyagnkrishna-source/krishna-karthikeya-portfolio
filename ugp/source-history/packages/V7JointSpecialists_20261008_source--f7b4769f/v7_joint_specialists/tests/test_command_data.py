import importlib.util
import sys
import unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'v7_joint_specialists'),str(ROOT/'v7feedback/.verification_deps'),str(ROOT/'v7legs_body')]

class CommandChecks(unittest.TestCase):
    def test_original_axis_prevents_rebasing_roundoff_leakage(self):
        _,_,window,_=self.api()
        crop=4.05;grid=np.array([15.75,15.76,15.77])
        p=dict(time=grid-crop,grid_original=grid,crop_start_original=np.array(crop),
               P=np.ones((3,4,12)),E=np.ones((3,4,3)),contact=np.ones((3,4)),B=np.ones((3,12)),U=np.ones((3,6)))
        original=np.array([4.05,15.758749999999996,np.nextafter(15.76,-np.inf),15.760000000000002])
        encoded=np.zeros((4,4,48));encoded[:,:,0]=np.array([0,10,30,20])[:,None]
        c=dict(time=original-crop,time_original=original,encoded=encoded)
        history,_,_,_=window(p,c,float(p['time'][1]),np.array([p['time'][2]]),.01)
        self.assertEqual(history[0,-1,34],30,'A command after the anchor must not enter observed history')
        _,_,commands,mask=window(p,c,float(p['time'][0]),np.array([p['time'][1]]),.01)
        self.assertEqual(int(mask[0,0].sum()),2,'A truly before-target command remains valid despite rounded relative time')
        self.assertTrue((commands[0,mask[0,0],0]==[10,30]).all())

    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('command_data'),'Command masks and causal windows missing')
        from command_data import encode_commands, sort_commands, window_arrays, check_targets
        return encode_commands,sort_commands,window_arrays,check_targets

    def fixture(self):
        times=np.arange(11,dtype=np.float64)/10
        p=dict(time=times,P=np.ones((11,4,12)),E=np.ones((11,4,3)),contact=np.ones((11,4)),B=np.ones((11,12)),U=np.ones((11,6)))
        c=dict(time=np.array([0,.25,.4,.49,.5,.6,.8]),encoded=np.ones((7,4,48)))
        return p,c

    def test_modes_mapping_inactive_and_unexpected(self):
        encode,*_=self.api()
        values=np.arange(72,dtype=float).reshape(1,12,6)
        values[0,0,1:]=np.nan
        modes=np.array([[10,12,13,99]*3]); mapping=np.arange(12)[::-1]
        e,audit=encode(values,modes,mapping)
        flat=e.reshape(1,12,16)
        self.assertTrue(np.isfinite(flat).all())
        np.testing.assert_array_equal(flat[0,-1,:6],[0,0,0,0,0,0])
        np.testing.assert_array_equal(flat[0,-1,6:12],[1,0,0,0,0,0])
        self.assertEqual(flat[0,0,15],1)
        self.assertTrue((flat[0,0,:12]==0).all())
        np.testing.assert_array_equal(flat[0,-2,6:12],[1,1,1,0,0,0])
        self.assertEqual(audit['unexpected_joint_commands'],3)

    def test_sort_duplicate_and_no_future_sensor_or_command(self):
        _,sort,window,_=self.api()
        t,x,diag=sort(np.array([.2,.1,.1]),np.array([[2],[10],[11]]))
        np.testing.assert_array_equal(x[:,0],[10,11,2]);self.assertEqual(diag['duplicates'],1)
        p,c=self.fixture(); a=window(p,c,.3,np.array([.5,.7]),.2)
        p['P'][4:]=999;p['B'][4:]=999
        b=window(p,c,.3,np.array([.5,.7]),.2)
        np.testing.assert_array_equal(a[0],b[0])
        # .49 is valid at .5; exactly .5 and .6 are unavailable to first horizon.
        selected=c['time'][(c['time']>=.3)&(c['time']<.7)]
        self.assertTrue(a[3][0,0,np.flatnonzero(selected==.49)[0]])
        self.assertFalse(a[3][0,0,np.flatnonzero(selected==.5)[0]])
        c['time']=np.array([0.,.8]);c['encoded']=np.ones((2,4,48))
        empty=window(p,c,.3,np.array([.5]),.2)
        self.assertFalse(empty[3].any())

    def test_misaligned_target_rejected(self):
        *_,check=self.api();p,_=self.fixture()
        check(p,.3,np.array([.5]),np.ones((1,4,9)),np.ones((1,4,3)))
        with self.assertRaises(ValueError): check(p,.3,np.array([.51]),np.ones((1,4,9)),np.ones((1,4,3)))

if __name__=='__main__':unittest.main()
