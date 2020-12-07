from datetime import datetime

import numpy as np


def test_import() -> None:
    from pytides2 import astro, constituent, nodal_corrections, tide


def test_Tide() -> None:
    from pytides2.tide import Tide

    data = np.genfromtxt(
        "test/data/water_level.csv",
        dtype=np.dtype([("datetime", datetime), ("water_level", float)]),
        delimiter=",",
    )
    water_datetimes = []
    water_levels = []
    for d in data:
        water_datetimes.append(
            datetime.strptime(d["datetime"].decode(), "%d/%m/%Y %H:%M:%S")
        )
        water_levels.append(d["water_level"])

    t, least_square = Tide.decompose(water_levels, water_datetimes)

    assert least_square is None
    assert t.model[0]["amplitude"] == 178.94736842105263
    assert t.model[0][2] == 0.0
    assert len(t.model) == 33
    assert t.formzahl == 0.3691950696609845 or t.formzahl == 0.38101230038937306
    assert t.type == "mixed (semidiurnal)"
