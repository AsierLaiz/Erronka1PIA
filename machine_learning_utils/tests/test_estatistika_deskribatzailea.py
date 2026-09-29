"""Probak. 'statistics' modulua PROBETAN bakarrik erabiltzen da emaitzak
alderatzeko; liburutegiaren kodean ez da erabiltzen."""
import doctest
import random
import statistics
import unittest

from machine_learning_utils import estatistika_deskribatzailea as ed


class TestFuntzioak(unittest.TestCase):
    def test_batez_bestekoa(self):
        self.assertEqual(ed.batez_bestekoa([2, 4, 6, 8]), 5.0)
        self.assertAlmostEqual(ed.batez_bestekoa([0.1, 0.2, 0.3]), 0.2)

    def test_mediana_bakoitia_bikoitia(self):
        self.assertEqual(ed.mediana([7, 1, 3]), 3)
        self.assertEqual(ed.mediana([4, 1, 3, 2]), 2.5)
        self.assertEqual(ed.mediana([5]), 5)

    def test_pertzentila(self):
        d = [15, 20, 35, 40, 50]
        self.assertEqual(ed.pertzentila(d, 0), 15)
        self.assertEqual(ed.pertzentila(d, 30), 20)
        self.assertEqual(ed.pertzentila(d, 40), 20)
        self.assertEqual(ed.pertzentila(d, 50), 35)
        self.assertEqual(ed.pertzentila(d, 100), 50)

    def test_bariantza_desbiderapena(self):
        d = [2, 4, 4, 4, 5, 5, 7, 9]
        self.assertEqual(ed.bariantza(d), 4.0)
        self.assertEqual(ed.desbiderapen_tipikoa(d), 2.0)
        self.assertAlmostEqual(ed.bariantza(d, lagina=True), statistics.variance(d))

    def test_laburpena(self):
        r = ed.laburpen_estatistikoa([1, 2, 3, 4, 5])
        self.assertEqual(r["minimoa"], 1)
        self.assertEqual(r["maximoa"], 5)
        self.assertEqual(r["p25"], 2)
        self.assertEqual(r["p75"], 4)
        self.assertEqual(len(r), 9)

    def test_alderaketa_statistics_ausazkoa(self):
        rnd = random.Random(42)
        for _ in range(200):
            d = [rnd.uniform(-100, 100) for _ in range(rnd.randint(2, 60))]
            self.assertAlmostEqual(ed.batez_bestekoa(d), statistics.fmean(d))
            self.assertAlmostEqual(ed.mediana(d), statistics.median(d))
            self.assertAlmostEqual(ed.bariantza(d), statistics.pvariance(d), places=6)
            self.assertAlmostEqual(ed.desbiderapen_tipikoa(d), statistics.pstdev(d), places=6)

    def test_ez_du_jatorrizkoa_aldatzen(self):
        d = [3, 1, 2]
        ed.mediana(d)
        ed.laburpen_estatistikoa(d)
        self.assertEqual(d, [3, 1, 2])

    def test_erroreak(self):
        with self.assertRaises(ValueError):
            ed.batez_bestekoa([])
        with self.assertRaises(TypeError):
            ed.mediana("abc")
        with self.assertRaises(TypeError):
            ed.bariantza([1, "2"])
        with self.assertRaises(TypeError):
            ed.batez_bestekoa([True, False])
        with self.assertRaises(ValueError):
            ed.pertzentila([1, 2], 101)
        with self.assertRaises(ValueError):
            ed.bariantza([1], lagina=True)

    def test_dokumentazioa(self):
        for f in (ed.batez_bestekoa, ed.mediana, ed.pertzentila, ed.bariantza,
                  ed.desbiderapen_tipikoa, ed.laburpen_estatistikoa):
            self.assertTrue(f.__doc__ and len(f.__doc__) > 50)

    def test_doctests(self):
        self.assertEqual(doctest.testmod(ed).failed, 0)

    def test_inportazioak(self):
        import ast, inspect
        arbola = ast.parse(inspect.getsource(ed))
        inportatuak = [(n.module, [a.name for a in n.names])
                       for n in ast.walk(arbola) if isinstance(n, ast.ImportFrom)]
        inportatuak += [([a.name for a in n.names]) for n in ast.walk(arbola) if isinstance(n, ast.Import)]
        self.assertEqual(inportatuak, [("math", ["ceil", "sqrt"])])


if __name__ == "__main__":
    unittest.main()
