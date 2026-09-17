"""Independent mathematical checks; run with python -m unittest discover -s tests."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis import load_returns, pca_covariance, mardia

class AnalysisChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x, cls.b, cls.info = load_returns()

    def test_eigenvalues_against_independent_svd(self):
        a = self.x.to_numpy()
        _, eig, _, _, _, _ = pca_covariance(a)
        singular = np.linalg.svd(a-a.mean(axis=0), compute_uv=False)
        np.testing.assert_allclose(eig, singular**2/(len(a)-1), rtol=1e-10)

    def test_portfolio_decomposition_against_observed_variance(self):
        a = self.x.to_numpy()
        _, _, _, _, var, contributions = pca_covariance(a)
        observed = np.var(a.mean(axis=1), ddof=1)
        self.assertAlmostEqual(var, observed, places=13)
        self.assertAlmostEqual(contributions.sum(), observed, places=13)

    def test_conditional_formula_against_regression_residuals(self):
        a, b = self.x.to_numpy(), self.b.to_numpy()
        joint = np.cov(np.column_stack([a,b]), rowvar=False)
        formula = joint[:-1,:-1] - np.outer(joint[:-1,-1],joint[:-1,-1])/joint[-1,-1]
        design = np.column_stack([np.ones(len(b)), b])
        residual = a - design @ np.linalg.lstsq(design,a,rcond=None)[0]
        np.testing.assert_allclose(formula,np.cov(residual,rowvar=False),atol=1e-14)

    def test_mardia_invariant_to_unit_changes(self):
        a=self.x.to_numpy(); original=mardia(a); scaled=mardia(a*100)
        self.assertAlmostEqual(original['b1'],scaled['b1'],places=8)
        self.assertAlmostEqual(original['b2'],scaled['b2'],places=8)

    def test_singular_input_is_rejected(self):
        a=self.x.to_numpy().copy(); a[:,1]=a[:,0]
        with self.assertRaises(ValueError): pca_covariance(a)

    def test_snapshot_shape_and_finite_values(self):
        self.assertEqual(self.x.shape[1],10)
        self.assertGreater(len(self.x),700)
        self.assertTrue(np.isfinite(self.x.to_numpy()).all())
        self.assertEqual(self.info['incomplete_return_rows_excluding_first'],0)

if __name__ == '__main__': unittest.main()
