import pytest

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput
from python_basics import (
    count_one_bits, count_vowels, has_unique_characters, is_balanced_number,
    mse, multiplicative_persistence, prime_factorization, pyramid,
)


def P(n):
    return PositiveIntegerInput(n)


# Задача 1
@pytest.mark.parametrize('text, expected', [
    ('hello', 2),
    ('AEIOU', 5),
    ('aEiOu', 5),
    ('rhythm', 0),
    ('', 0),
    ('Programming', 3),
])
def test_count_vowels(text, expected):
    assert count_vowels(TextInput(text)) == expected


# Задача 2
@pytest.mark.parametrize('text, expected', [
    ('abc', True),
    ('abca', False),
    ('', True),
    ('a', True),
    ('aA', True),     # регистр различается
    ('  ', False),    # повторяющийся пробел
])
def test_has_unique_characters(text, expected):
    assert has_unique_characters(TextInput(text)) is expected


# Задача 3
@pytest.mark.parametrize('n, expected', [
    (1, 1), (2, 1), (7, 3), (8, 1), (255, 8), (1023, 10), (2 ** 40, 1),
])
def test_count_one_bits(n, expected):
    assert count_one_bits(P(n)) == expected


# Задача 4
@pytest.mark.parametrize('n, expected', [
    (39, 3), (4, 0), (999, 4), (10, 1), (25, 2), (1, 0), (277777788888899, 11),
])
def test_multiplicative_persistence(n, expected):
    assert multiplicative_persistence(P(n)) == expected


# Задача 5
def test_mse_basic():
    assert mse(VectorPairInput([1, 2, 3], [1, 2, 3])) == 0
    assert mse(VectorPairInput([1, 2, 3], [2, 4, 6])) == pytest.approx(14 / 3)
    assert mse(VectorPairInput([5], [2])) == 9
    assert mse(VectorPairInput([-1, 1], [1, -1])) == 4


def test_mse_errors():
    with pytest.raises(ValueError):
        mse(VectorPairInput([1, 2], [1]))
    with pytest.raises(ValueError):
        mse(VectorPairInput([], []))


# Задача 6
@pytest.mark.parametrize('n, expected', [
    (86240, '(2**5)(5)(7**2)(11)'),
    (2, '(2)'),
    (97, '(97)'),
    (8, '(2**3)'),
    (12, '(2**2)(3)'),
    (7775460, '(2**2)(3**3)(5)(7)(11**2)(17)'),
    (2 * 1000003, '(2)(1000003)'),
])
def test_prime_factorization(n, expected):
    assert prime_factorization(P(n)) == expected


def test_prime_factorization_one():
    assert prime_factorization(P(1)) == ''


# Задача 7
@pytest.mark.parametrize('n, expected', [
    (1, 1), (5, 2), (14, 3), (30, 4), (55, 5), (338350, 100),
    (2, 'It is impossible'), (4, 'It is impossible'), (15, 'It is impossible'),
])
def test_pyramid(n, expected):
    assert pyramid(P(n)) == expected


# Задача 8
@pytest.mark.parametrize('n, expected', [
    (1234006, True),
    (123456, False),
    (7, True),        # одна цифра: обе суммы пустые
    (12, True),       # две цифры: обе средние, суммы пустые
    (959, True),
    (13, True),
    (1230, False),    # слева 1, справа 0
    (56239814, True),  # 5+6+2 == 8+1+4
    (12345, False),
])
def test_is_balanced_number(n, expected):
    assert is_balanced_number(P(n)) is expected
