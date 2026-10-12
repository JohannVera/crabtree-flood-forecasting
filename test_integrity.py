"""Checks that future observations cannot alter earlier hydrologic inputs."""
import unittest
import numpy as np
import pandas as pd
from forecast import features

class Integrity(unittest.TestCase):
    def test_causality(self):
        t=pd.date_range('2020-01-01',periods=600,freq='5min',tz='UTC')
        d=pd.DataFrame({'level_ft':np.arange(600.),'rain_in':np.ones(600)},index=t)
        original=features(d)
        d.loc[t[400]:,['level_ft','rain_in']]=99999
        pd.testing.assert_frame_equal(original.loc[:t[399]],features(d).loc[:t[399]])
    def test_missing_rain_not_zero(self):
        t=pd.date_range('2020-01-01',periods=30,freq='5min',tz='UTC')
        d=pd.DataFrame({'level_ft':1.,'rain_in':1.},index=t)
        d.loc[t[15],'rain_in']=np.nan
        self.assertTrue(np.isnan(features(d).loc[t[16],'rain_sum_12']))
        self.assertGreater(features(d).loc[t[16],'rain_missing_fraction_12'],0)

if __name__=='__main__':unittest.main()
