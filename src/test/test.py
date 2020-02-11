from src.main.solution import *
from unittest import TestCase
import numpy as np


class TestMultiLayerPerceptron(TestCase):

    def test_softmax_vector(self):
        mlp = NN()
        x = np.array([2, 4, 2])
        expected = np.array([0.1065069789192, 0.7869860421616, 0.1065069789192])
        actual = mlp.softmax(x)
        np.testing.assert_array_almost_equal(actual, expected, 4)

    def test_softmax_matrix(self):
        mlp = NN()
        x = np.array([[1, 2, 3], [5, 6, 4]])
        expected = np.array([[0.090030573170381, 0.2447284710548, 0.66524095577482], [0.2447284710548, 0.66524095577482, 0.090030573170381]])
        actual = mlp.softmax(x)
        np.testing.assert_array_almost_equal(actual, expected, 4)


