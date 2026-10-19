"""Заготовки задач на базовый Python."""

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    vowels = set('aeiou')
    return sum(1 for ch in text.lower() if ch in vowels)


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(set(text)) == len(text)


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count('1')


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    if len(predicted) != len(expected):
        raise ValueError('Векторы должны быть одинаковой длины')
    if len(predicted) == 0:
        raise ValueError('Векторы не должны быть пустыми')
    total = sum((x - y) ** 2 for x, y in zip(predicted, expected))
    return total / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    parts = []
    divisor = 2
    while divisor * divisor <= number:
        power = 0
        while number % divisor == 0:
            number //= divisor
            power += 1
        if power == 1:
            parts.append(f'({divisor})')
        elif power > 1:
            parts.append(f'({divisor}**{power})')
        divisor += 1
    if number > 1:
        parts.append(f'({number})')
    return ''.join(parts)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    total = 0
    k = 0
    while total < cube_count:
        k += 1
        total += k * k
    if total == cube_count and k > 0:
        return k
    return 'It is impossible'


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    digits = [int(d) for d in str(number)]
    n = len(digits)
    half = (n - 1) // 2  # сколько цифр слева от середины
    left = digits[:half]
    right = digits[n - half:] if half > 0 else []
    return sum(left) == sum(right)
