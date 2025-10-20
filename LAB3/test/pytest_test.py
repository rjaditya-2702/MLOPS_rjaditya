import pytest
import sys
import os

#set path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from calculator import fun1, fun2, fun3, fun4

# --------------------------
# Tests for fun1 (Addition)
# --------------------------
@pytest.mark.parametrize("x, y, expected", [
    (2, 3, 5),
    (-2, 3, 1),
    (3, -2, 1),
    (-3, -5, -8),
    (0, 0, 0),
])
def test_fun1_addition(x, y, expected):
    assert fun1(x, y) == expected


def test_fun1_invalid_types():
    with pytest.raises(TypeError):
        fun1("a", 2)
    with pytest.raises(TypeError):
        fun1(2, "b")
    with pytest.raises(TypeError):
        fun1("a", "b")


# --------------------------
# Tests for fun2 (Subtraction)
# --------------------------
@pytest.mark.parametrize("x, y, expected", [
    (5, 3, -2),    # y - x = -2
    (3, 5, 2),
    (-2, -3, -1),
    (0, 0, 0),
    (10, -5, -15),
])
def test_fun2_subtraction(x, y, expected):
    assert fun2(x, y) == expected


def test_fun2_invalid_types():
    with pytest.raises(TypeError):
        fun2("x", 5)
    with pytest.raises(TypeError):
        fun2(5, [1, 2])


# --------------------------
# Tests for fun3 (Multiplication)
# --------------------------
@pytest.mark.parametrize("x, y, expected", [
    (2, 3, 6),
    (-2, 3, -6),
    (-3, -5, 15),
    (0, 10, 0),
])
def test_fun3_multiplication(x, y, expected):
    assert fun3(x, y) == expected


def test_fun3_invalid_types():
    with pytest.raises(TypeError):
        fun3(3, "b")
    with pytest.raises(TypeError):
        fun3("a", "b")
    with pytest.raises(TypeError):
        fun3(None, 2)


# --------------------------
# Tests for fun4 (Combined)
# --------------------------
@pytest.mark.parametrize("x, y", [
    (2, 3),
    (-1, 2),
    (-4, -5),
    (0, 0),
])
def test_fun4_combined_math(x, y):
    """Verify fun4 performs correct combined math."""
    expected = fun1(x, y) + fun2(x, y) + fun3(x, y)
    assert fun4(x, y) == expected


def test_fun4_invalid_types():
    with pytest.raises(TypeError):
        fun4(2, "x")
    with pytest.raises(TypeError):
        fun4("a", "b")


# --------------------------
# Consistency and sanity checks
# --------------------------
def test_fun4_is_consistent_with_individual_functions():
    x, y = 4, 5
    expected = fun1(x, y) + fun2(x, y) + fun3(x, y)
    assert fun4(x, y) == expected


def test_zero_behavior():
    assert fun1(0, 0) == 0
    assert fun2(0, 0) == 0
    assert fun3(0, 0) == 0
    assert fun4(0, 0) == 0
