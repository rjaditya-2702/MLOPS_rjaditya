import unittest
import sys
import os

# Add src folder to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from calculator import fun1, fun2, fun3, fun4


class TestCalculator(unittest.TestCase):

    # --------------------------
    # fun1 tests
    # --------------------------
    def test_fun1_addition(self):
        self.assertEqual(fun1(2, 3), 5)
        self.assertEqual(fun1(-2, 3), 1)
        self.assertEqual(fun1(3, -2), 1)
        self.assertEqual(fun1(-3, -5), -8)
        self.assertEqual(fun1(0, 0), 0)

    def test_fun1_invalid_types(self):
        with self.assertRaises(TypeError):
            fun1("a", 2)
        with self.assertRaises(TypeError):
            fun1(2, "b")
        with self.assertRaises(TypeError):
            fun1("a", "b")

    # --------------------------
    # fun2 tests
    # --------------------------
    def test_fun2_subtraction(self):
        self.assertEqual(fun2(5, 3), -2)
        self.assertEqual(fun2(3, 5), 2)
        self.assertEqual(fun2(-2, -3), -1)
        self.assertEqual(fun2(0, 0), 0)
        self.assertEqual(fun2(10, -5), -15)

    def test_fun2_invalid_types(self):
        with self.assertRaises(TypeError):
            fun2("x", 5)
        with self.assertRaises(TypeError):
            fun2(5, [1, 2])

    # --------------------------
    # fun3 tests
    # --------------------------
    def test_fun3_multiplication(self):
        self.assertEqual(fun3(2, 3), 6)
        self.assertEqual(fun3(-2, 3), -6)
        self.assertEqual(fun3(-3, -5), 15)
        self.assertEqual(fun3(0, 10), 0)

    def test_fun3_invalid_types(self):
        with self.assertRaises(TypeError):
            fun3(3, "b")
        with self.assertRaises(TypeError):
            fun3("a", "b")
        with self.assertRaises(TypeError):
            fun3(None, 2)

    # --------------------------
    # fun4 tests
    # --------------------------
    def test_fun4_combined_math(self):
        for x, y in [(2, 3), (-1, 2), (-4, -5), (0, 0)]:
            expected = fun1(x, y) + fun2(x, y) + fun3(x, y)
            self.assertEqual(fun4(x, y), expected)

    def test_fun4_invalid_types(self):
        with self.assertRaises(TypeError):
            fun4(2, "x")
        with self.assertRaises(TypeError):
            fun4("a", "b")

    def test_fun4_consistency(self):
        x, y = 4, 5
        expected = fun1(x, y) + fun2(x, y) + fun3(x, y)
        self.assertEqual(fun4(x, y), expected)

    def test_zero_behavior(self):
        self.assertEqual(fun1(0, 0), 0)
        self.assertEqual(fun2(0, 0), 0)
        self.assertEqual(fun3(0, 0), 0)
        self.assertEqual(fun4(0, 0), 0)


if __name__ == "__main__":
    unittest.main()
