import sys

VALUES = {
    'א': 1, 'ב': 2, 'ג': 3, 'ד': 4, 'ה': 5, 'ו': 6, 'ז': 7, 'ח': 8, 'ט': 9,
    'י': 10, 'כ': 20, 'ל': 30, 'מ': 40, 'נ': 50, 'ס': 60, 'ע': 70, 'פ': 80,
    'צ': 90, 'ק': 100, 'ר': 200, 'ש': 300, 'ת': 400,
}

UNITS = {1: 'א', 2: 'ב', 3: 'ג', 4: 'ד', 5: 'ה', 6: 'ו', 7: 'ז', 8: 'ח', 9: 'ט'}
TENS = {10: 'י', 20: 'כ', 30: 'ל', 40: 'מ', 50: 'נ', 60: 'ס', 70: 'ע', 80: 'פ', 90: 'צ'}
HUNDREDS = {100: 'ק', 200: 'ר', 300: 'ש', 400: 'ת'}
FINALS = {'ך': 'כ', 'ם': 'מ', 'ן': 'נ', 'ף': 'פ', 'ץ': 'צ'}


def canonical_letters(n):
    """Build the one-and-only canonical Hebrew numeral letters for n."""
    thousands, remainder = divmod(n, 1000)
    letters = UNITS.get(thousands, '') if thousands else ''

    hundreds_val = (remainder // 100) * 100
    remainder_100 = remainder % 100
    h = hundreds_val
    while h > 0:
        if h >= 400:
            letters += 'ת'
            h -= 400
        else:
            letters += HUNDREDS[h]
            h = 0

    if remainder_100 == 15:
        letters += 'טו'
    elif remainder_100 == 16:
        letters += 'טז'
    else:
        tens = (remainder_100 // 10) * 10
        ones = remainder_100 % 10
        if tens:
            letters += TENS[tens]
        if ones:
            letters += UNITS[ones]

    return letters


def word_value(word):
    """Return the gematria sum of a word only if it's a valid, canonically
    written Hebrew numeral; otherwise return -1."""
    normalized = ''.join(FINALS.get(ch, ch) for ch in word)
    letters = [ch for ch in normalized if ch in VALUES]
    if not letters:
        return -1
    total = sum(VALUES[ch] for ch in letters)
    if ''.join(letters) == canonical_letters(total):
        return total
    return -1


def hebrew_to_values(text):
    """Return a list of gematria sums, one per valid numeral word."""
    return [v for word in text.split() if (v := word_value(word)) is not None]


if __name__ == "__main__":
    while True:
        text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter Hebrew text: ")
        print(hebrew_to_values(text))
        if len(sys.argv) > 1:
            break