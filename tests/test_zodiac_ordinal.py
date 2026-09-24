"""Finite metric and synthetic-ordinal controls for ZODIAC-0001."""

from voynich.zodiac_ordinal import cosine, make_idf, ngrams, rank_fraction, vector


def test_ngram_cosine_and_ties() -> None:
    idf = make_idf(["abcz", "abcy", "xyqz"])
    assert "^a" in ngrams("abcz") and "cz$" in ngrams("abcz")
    assert cosine(vector("abcz", idf), vector("abcz", idf)) == 1.0
    assert cosine(vector("mmmm", idf), vector("xyqz", idf)) == 0.0
    assert rank_fraction([1.0] * 30, 12) == (0.5, 15.5)


def test_known_ordinal_codes_rank_above_wrong_positions() -> None:
    codes = [chr(97 + i // 26) + chr(97 + i % 26) for i in range(30)]
    train = ["x" + code + "y" for code in codes]
    idf = make_idf(train)
    prototypes = [vector(word, idf) for word in train]
    fractions = []
    for index, code in enumerate(codes):
        query = vector("z" + code + "w", idf)
        scores = [cosine(query, prototype) for prototype in prototypes]
        fractions.append(rank_fraction(scores, index)[0])
    assert sum(fractions) / 30 > 0.9
