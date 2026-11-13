"""不加载模型的数值单元测试：python3 -m unittest discover -s 此目录"""
import unittest
import numpy as np
from run import cka, auc
class Tests(unittest.TestCase):
    def test_standardized_cka_scale(self):
        r=np.random.default_rng(0);x=r.normal(size=(128,20));s=np.exp(r.uniform(-4,4,20))
        self.assertAlmostEqual(cka(x,x*s,True),1.,places=12)
        self.assertLess(cka(x,x*s),.9)
    def test_auc(self):
        self.assertEqual(auc([1,2],[0,0]),1.);self.assertEqual(auc([1],[1]),.5)
    def test_pairing(self):
        r=np.random.default_rng(0);x=r.normal(size=(128,20))
        self.assertLess(cka(x,x[r.permutation(128)],True),.4)
if __name__=='__main__':unittest.main()
