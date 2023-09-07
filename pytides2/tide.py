from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import count, takewhile
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
from scipy.optimize import fsolve, leastsq

from pytides2.constituent import BaseConstituent, CompoundConstituent

from . import constituent
from .astro import AstronomicalParameter, astro

d2r: float = np.pi / 180.0
r2d: float = 180.0 / np.pi


@dataclass
class TideDataSource:
    times: List[datetime]
    water_levels: np.ndarray


class Tide(object):
    dtype = np.dtype([("constituent", object), ("amplitude", float), ("phase", float)])

    def __init__(
        self,
        constituents: Optional[List[AstronomicalParameter]] = None,
        amplitudes: Optional[List[float]] = None,
        phases: Optional[List[float]] = None,
        model: np.ndarray = None,
        radians: bool = False,
        source: Optional[TideDataSource] = None,
    ) -> None:
        """
        Initialise a tidal model. Provide constituents, amplitudes and phases OR a model.
        Arguments:
        constituents -- list of constituents used in the model.
        amplitudes -- list of amplitudes corresponding to constituents
        phases -- list of phases corresponding to constituents
        model -- an ndarray of type Tide.dtype representing the constituents, amplitudes and phases.
        radians -- boolean representing whether phases are in radians (default False)
        """
        if constituents is not None and amplitudes is not None and phases is not None:
            if len(constituents) == len(amplitudes) == len(phases):
                model = np.zeros(len(phases), dtype=Tide.dtype)
                model["constituent"] = np.array(constituents)
                model["amplitude"] = np.array(amplitudes)
                model["phase"] = np.array(phases)
            else:
                raise ValueError(
                    "Constituents, amplitudes and phases should all be arrays of equal length."
                )
        elif model is not None:
            if not model.dtype == Tide.dtype:
                raise ValueError("Model must be a numpy array with dtype == Tide.dtype")
        else:
            raise ValueError(
                "Must be initialized with constituents, amplitudes and phases; or a model."
            )
        if radians:
            model["phase"] = r2d * model["phase"]
        self.model: np.ndarray = model[:]
        self.normalize()

        self.source = source
        self.model_dict: Dict[str, Dict[str, float]] = {}
        for m, a, p in self.model:
            self.model_dict[m] = {"amplitude": a, "phase": p}

    def prepare(
        self, *args, **kwargs
    ) -> Tuple[np.ndarray, List[float], List[float], np.ndarray]:
        return Tide._prepare(self.model["constituent"], *args, **kwargs)

    @staticmethod
    def _prepare(
        constituents: Union[List[BaseConstituent], List[CompoundConstituent]],
        t0: Union[datetime, List[datetime]],
        t: Union[None, datetime, List[datetime]] = None,
        radians: bool = True,
    ) -> Tuple[np.ndarray, List[float], List[float], np.ndarray]:
        """
        Return constituent speed and equilibrium argument at a given time, and constituent node factors at given times.
        Arguments:
        constituents -- list of constituents to prepare
        t0 -- time at which to evaluate speed and equilibrium argument for each constituent
        t -- list of times at which to evaluate node factors for each constituent (default: t0)
        radians -- whether to return the angular arguments in radians or degrees (default: True)
        """
        # The equilibrium argument is constant and taken at the beginning of the
        # time series (t0).  The speed of the equilibrium argument changes very
        # slowly, so again we take it to be constant over any length of data. The
        # node factors change more rapidly.
        if isinstance(t0, Iterable):
            t0 = t0[0]
        if t is None:
            t = [t0]
        if not isinstance(t, Iterable):
            t = [t]
        a0 = astro(t0)
        a = [astro(t_i) for t_i in t]

        # For convenience give u, V0 (but not speed!) in [0, 360)
        V0: np.ndarray = np.array([c.V(a0) for c in constituents])[:, np.newaxis]
        speed: np.ndarray = np.array([c.speed(a0) for c in constituents])[:, np.newaxis]
        u: List[float] = [
            np.mod(np.array([c.u(a_i) for c in constituents])[:, np.newaxis], 360.0)
            for a_i in a
        ]
        f: List[float] = [
            np.mod(np.array([c.f(a_i) for c in constituents])[:, np.newaxis], 360.0)
            for a_i in a
        ]

        if radians:
            speed *= d2r
            V0 *= d2r
            u = [d2r * each for each in u]
        return speed, u, f, V0

    def plot(self) -> None:
        if self.source is not None:
            pass

    def at(self, t: List[datetime]) -> np.ndarray:
        """Calculate tidal height at given times

        Parameters
        ----------
        t : List[datetime]
            array of times at which to evaluate the tidal height

        Returns
        -------
        np.ndarray
            the modelled tidal height at given times.
        """
        t0 = t[0]
        hours = self._hours(t0, t)
        partition = 240.0
        t = self._partition(hours, partition)
        times = self._times(t0, [(i + 0.5) * partition for i in range(len(t))])
        speed, u, f, V0 = self.prepare(t0, times, radians=True)
        H = self.model["amplitude"][:, np.newaxis]
        p = d2r * self.model["phase"][:, np.newaxis]

        return np.concatenate(
            [
                Tide._tidal_series(t_i, H, p, speed, u_i, f_i, V0)
                for t_i, u_i, f_i in zip(t, u, f)
            ]
        )

    def highs(self, *args) -> Generator[float, None, None]:
        """yielding only the high tides

        Yields
        -------
        Generator[float, None, None]
            see Tide.extrema()
        """
        for t in filter(lambda e: e[2] == "H", self.extrema(*args)):
            yield t

    def lows(self, *args):
        """
        Generator yielding only the low tides.
        Arguments:
        see Tide.extrema()
        """
        for t in filter(lambda e: e[2] == "L", self.extrema(*args)):
            yield t

    def form_number(self) -> float:
        """
        Returns the model's form number, a helpful heuristic for classifying tides.
        """
        k1, o1, m2, s2 = (
            np.extract(self.model["constituent"] == c, self.model["amplitude"])
            for c in [
                constituent._K1,
                constituent._O1,
                constituent._M2,
                constituent._S2,
            ]
        )
        return float(((k1 + o1) / (m2 + s2))[0])

    @property
    def formzahl(self) -> float:
        return self.form_number()

    @property
    def classify(self) -> str:
        """
        Classify the tide according to its form number
        """
        form = self.formzahl
        if 0 <= form <= 0.25:
            return "semidiurnal"
        elif 0.25 < form <= 1.5:
            return "mixed (semidiurnal)"
        elif 1.5 < form <= 3.0:
            return "mixed (diurnal)"
        else:
            return "diurnal"

    @property
    def type(self) -> str:
        return self.classify

    def extrema(
        self, t0: datetime, t1: Optional[datetime] = None, partition: float = 2400.0
    ) -> Generator[Any, None, None]:
        """
        A generator for high and low tides.
        Arguments:
        t0 -- time after which extrema are sought
        t1 -- optional time before which extrema are sought (if not given, the generator is infinite)
        partition -- number of hours for which we consider the node factors to be constant (default: 2400.0)
        """
        if t1:
            # yield from in python 3.4
            for e in takewhile(lambda t: t[0] < t1, self.extrema(t0)):
                yield e
        else:
            # We assume that extrema are separated by at least delta hours
            delta: np.ndarray = np.amin(
                [
                    90.0 / c.speed(astro(t0))
                    for c in self.model["constituent"]
                    if not c.speed(astro(t0)) == 0
                ]
            )
            # We search for stationary points from offset hours before t0 to
            # ensure we find any which might occur very soon after t0.
            offset = 24.0
            partitions = (Tide._times(t0, i * partition) for i in count()), (
                Tide._times(t0, i * partition) for i in count(1)
            )

            # We'll overestimate to be on the safe side;
            # values outside (start,end) won't get yielded.
            interval_count = int(np.ceil((partition + offset) / delta)) + 1
            amplitude = self.model["amplitude"][:, np.newaxis]
            phase = d2r * self.model["phase"][:, np.newaxis]

            for start, end in zip(*partitions):
                speed, [u], [f], V0 = self.prepare(
                    start, Tide._times(start, 0.5 * partition)
                )
                # These derivatives don't include the time dependence of u or f,
                # but these change slowly.
                def d(t: np.ndarray) -> Union[np.ndarray, float]:
                    return np.sum(
                        -speed * amplitude * f * np.sin(speed * t + (V0 + u) - phase),
                        axis=0,
                    )

                def d2(t: np.ndarray) -> Union[np.ndarray, float]:
                    return np.sum(
                        -(speed ** 2.0)
                        * amplitude
                        * f
                        * np.cos(speed * t + (V0 + u) - phase),
                        axis=0,
                    )

                # We'll overestimate to be on the safe side;
                # values outside (start,end) won't get yielded.
                intervals = (delta * i - offset for i in range(interval_count)), (
                    delta * (i + 1) - offset for i in range(interval_count)
                )
                for a, b in zip(*intervals):
                    if d(a) * d(b) < 0:
                        extrema: np.ndarray = fsolve(d, (a + b) / 2.0, fprime=d2)[0]
                        time = Tide._times(start, extrema)
                        [height] = self.at([time])
                        hilo = "H" if d2(extrema) < 0 else "L"
                        if start < time < end:
                            yield (time, height, hilo)

    @staticmethod
    def _hours(
        t0: datetime, t: Union[datetime, List[datetime]]
    ) -> Union[List[datetime], np.ndarray]:
        """
        Return the hourly offset(s) of a (list of) time from a given time.
        Arguments:
        t0 -- time from which offsets are sought
        t -- times to find hourly offsets from t0.
        """
        if not isinstance(t, Iterable):
            return Tide._hours(t0, [t])[0]
        elif isinstance(t[0], datetime):
            return np.array([(ti - t0).total_seconds() / 3600.0 for ti in t])
        else:
            return t

    @staticmethod
    def _partition(
        hours: np.ndarray, partition: float = 3600.0
    ) -> List[Union[datetime, float]]:
        """
        Partition a sorted list of numbers (or in this case hours).
        Arguments:
        hours -- sorted ndarray of hours.
        partition -- maximum partition length (default: 3600.0)
        """
        partition = float(partition)
        relative: np.ndarray = hours - hours[0]
        total_partitions: int = np.ceil(
            relative[-1] / partition + 10 * np.finfo(float).eps
        ).astype("int")
        return [
            hours[np.floor(np.divide(relative, partition)) == i]
            for i in range(total_partitions)
        ]

    @staticmethod
    def _times(
        t0: datetime, hours: Union[datetime, float, List[float], List[datetime]]
    ) -> np.ndarray:
        """
        Return a (list of) datetime(s) given an initial time and an (list of) hourly offset(s).
        Arguments:
        t0 -- initial time
        hours -- hourly offsets from t0
        """
        if not isinstance(hours, Iterable):
            return Tide._times(t0, [hours])[0]
        elif not isinstance(hours[0], datetime):
            return np.array([t0 + timedelta(hours=h) for h in hours])
        else:
            return np.array(hours)

    @staticmethod
    def _tidal_series(
        t: float,
        amplitude: np.ndarray,
        phase: np.ndarray,
        speed: np.ndarray,
        u: float,
        f: float,
        V0: np.ndarray,
    ) -> Union[float, Any]:
        return np.sum(amplitude * f * np.cos(speed * t + (V0 + u) - phase), axis=0)

    def normalize(self) -> None:
        """
        Adapt self.model so that amplitudes are positive and phases are in [0,360) as per convention
        """
        for i, (_, amplitude, phase) in enumerate(self.model):
            if amplitude < 0:
                self.model["amplitude"][i] = -amplitude
                self.model["phase"][i] = phase + 180.0
            self.model["phase"][i] = np.mod(self.model["phase"][i], 360.0)

    @classmethod
    def decompose(
        cls,
        heights: np.ndarray,
        t: Optional[np.ndarray] = None,
        t0: Optional[datetime] = None,
        interval: Optional[float] = None,
        constituents: List[
            Union[BaseConstituent, CompoundConstituent]
        ] = constituent.noaa,
        initial: Optional[np.ndarray] = None,
        n_period: Union[int, float] = 2,
        callback: Optional[Callable[..., Any]] = None,
        full_output: bool = False,
    ) -> Tuple[
        "Tide", Optional[Tuple[np.ndarray, np.ndarray, Dict[str, Any], str, int]]
    ]:
        """
        Return an instance of Tide which has been fitted to a series of tidal observations.
        Arguments:
        It is not necessary to provide t0 or interval if t is provided.
        heights -- ndarray of tidal observation heights
        t -- ndarray of tidal observation times
        t0 -- datetime representing the time at which heights[0] was recorded
        interval -- hourly interval between readings
        constituents -- list of constituents to use in the fit (default: constituent.noaa)
        initial -- optional Tide instance to use as first guess for least squares solver
        n_period -- only include constituents which complete at least this many periods (default: 2)
        callback -- optional function to be called at each iteration of the solver
        full_output -- whether to return the output of scipy's leastsq solver (default: False)
        """
        if t is not None:
            if isinstance(t[0], datetime):
                hours = Tide._hours(t[0], t)
                t0 = t[0]
            elif t0 is not None:
                hours = t
            else:
                raise ValueError(
                    "t can be an array of datetimes, or an array "
                    "of hours since t0 in which case t0 must be "
                    "specified."
                )
        elif interval is not None:
            hours = np.arange(len(heights)) * interval
        else:
            raise ValueError(
                "Must provide t(datetimes), or t(hours) and "
                "t0(datetime), or interval(hours) and t0(datetime) "
                "so that each height can be identified with an "
                "instant in time."
            )

        if t0 is not None:
            # Remove duplicate constituents (those which travel at exactly the same
            # speed, irrespective of phase)
            constituents = list(OrderedDict.fromkeys(constituents))

            # No need for least squares to find the mean water level constituent z0,
            # work relative to mean
            constituents = [c for c in constituents if not c == constituent._Z0]
            z0: float = np.mean(heights)
            heights = heights - z0

            # Only analyse frequencies which complete at least n_period cycles over
            # the data period.
            constituents = [
                c
                for c in constituents
                if 360.0 * n_period < hours[-1] * c.speed(astro(t0))
            ]
            n = len(constituents)

            sort: np.ndarray = np.argsort(hours)
            hours = hours[sort]
            heights = heights[sort]

            # We partition our time/height data into intervals over which we consider
            # the values of u and f to assume a constant value (that is, their true
            # value at the midpoint of the interval).  Constituent
            # speeds change much more slowly than the node factors, so we will
            # consider these constant and equal to their speed at t0, regardless of
            # the length of the time series.

            partition = 240.0

            t = Tide._partition(hours, partition)
            times = Tide._times(t0, [(i + 0.5) * partition for i in range(len(t))])

            speed, u, f, V0 = Tide._prepare(constituents, t0, times, radians=True)

            # Residual to be minimised by variation of parameters (amplitudes, phases)
            def residual(hp: np.ndarray) -> np.ndarray:
                H, p = hp[:n, np.newaxis], hp[n:, np.newaxis]
                s = np.concatenate(
                    [
                        Tide._tidal_series(t_i, H, p, speed, u_i, f_i, V0)
                        for t_i, u_i, f_i in zip(t, u, f)
                    ]
                )
                res = heights - s
                if callback:
                    callback(res)
                return res

            # Analytic Jacobian of the residual - this makes solving significantly
            # faster than just using gradient approximation, especially with many
            # measurements / constituents.
            def D_residual(hp: np.ndarray) -> np.ndarray:
                H, p = hp[:n, np.newaxis], hp[n:, np.newaxis]
                ds_dH = np.concatenate(
                    [
                        f_i * np.cos(speed * t_i + u_i + V0 - p)
                        for t_i, u_i, f_i in zip(t, u, f)
                    ],
                    axis=1,
                )

                ds_dp = np.concatenate(
                    [
                        H * f_i * np.sin(speed * t_i + u_i + V0 - p)
                        for t_i, u_i, f_i in zip(t, u, f)
                    ],
                    axis=1,
                )

                return np.append(-ds_dH, -ds_dp, axis=0)

            # Initial guess for solver, haven't done any analysis on this since the
            # solver seems to converge well regardless of the initial guess We do
            # however scale the initial amplitude guess with some measure of the
            # variation
            amplitudes: np.ndarray = np.ones(n) * (
                np.sqrt(np.dot(heights, heights)) / len(heights)
            )
            phases: np.ndarray = np.ones(n)

            if initial:
                for (c0, amplitude, phase) in initial.model:
                    for i, c in enumerate(constituents):
                        if c0 == c:
                            amplitudes[i] = amplitude
                            phases[i] = d2r * phase

            initial = np.append(amplitudes, phases)

            lsq: Tuple[np.ndarray, np.ndarray, Dict[str, Any], str, int] = leastsq(
                residual, initial, Dfun=D_residual, col_deriv=True, ftol=1e-7
            )

            model: np.ndarray = np.zeros(1 + n, dtype=cls.dtype)
            model[0] = (constituent._Z0, z0, 0)
            model[1:]["constituent"] = constituents[:]
            model[1:]["amplitude"] = lsq[0][:n]
            model[1:]["phase"] = lsq[0][n:]

            if full_output:
                return (
                    cls(
                        model=model,
                        radians=True,
                        source=TideDataSource(times=t, water_levels=heights),
                    ),
                    lsq,
                )
            return (
                cls(
                    model=model,
                    radians=True,
                    source=TideDataSource(times=t, water_levels=heights),
                ),
                None,
            )
        else:
            raise ValueError(
                "Must provide t(datetimes), or t(hours) and "
                "t0(datetime), or interval(hours) and t0(datetime) "
                "so that each height can be identified with an "
                "instant in time."
            )
