def parse_timespan(runtime: str) -> int:
    multiplicators = dict(h=60, m=1)
    multiplicators = [
        multiplicators[postfix] for postfix in multiplicators if runtime[-1] == postfix
    ]
    assert len(multiplicators) == 1

    return int(runtime[:-1]) * multiplicators[0]


def parse_byte_count(count: str) -> int:
    multiplicators = dict(GB=1000, MB=1)
    multiplicators = [
        multiplicators[postfix] for postfix in multiplicators if count[-2:] == postfix
    ]
    assert len(multiplicators) == 1

    return int(count[:-2]) * multiplicators[0]