"""Symmetric dark-forest model: exact individual events and a mean-field ODE."""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class Parameters:
    side: int = 10
    initial: int = 20
    birth: float = 0.001
    collapse: float = 0.01
    search: float = 1.0
    discovery: float = 0.5
    duration: float = 500.0
    sample_interval: float = 0.5

    @property
    def capacity(self):
        return self.side ** 2

    def __post_init__(self):
        if self.side < 2 or not 0 <= self.initial <= self.capacity:
            raise ValueError("Use side >= 2 and an initial count within capacity.")
        if min(self.birth, self.collapse, self.search) < 0:
            raise ValueError("Event rates must be nonnegative.")
        if not 0 <= self.discovery <= 1:
            raise ValueError("Discovery probability must be between zero and one.")
        if self.duration <= 0 or self.sample_interval <= 0:
            raise ValueError("Duration and sampling interval must be positive.")


@dataclass(frozen=True)
class Event:
    time: float
    kind: str
    source: int
    target: int
    source_id: int
    target_id: int
    count_before: int
    count_after: int


@dataclass
class Trial:
    parameters: Parameters
    seed: int
    initial_sites: list[int]
    events: list[Event]
    sample_times: list[float]
    sample_counts: list[int]
    final_count: int
    births: int
    collapses: int
    attacks: int
    searches: int
    empty_time: float


def simulate(parameters: Parameters, seed: int) -> Trial:
    """Gillespie events include all search attempts, including unsuccessful ones.

    Positions are fixed. A search chooses uniformly from all other stars;
    visibility is tested only if the selected star contains another civilization.
    IDs are unique to a civilization, even when a star is occupied again later.
    """
    p = parameters
    rng = random.Random(seed)
    initial_sites = sorted(rng.sample(range(p.capacity), p.initial))
    alive = {site: index for index, site in enumerate(initial_sites)}
    next_id = p.initial
    events = []
    counts = {"birth": 0, "collapse": 0, "attack": 0, "search": 0}
    time = 0.0
    empty_time = 0.0

    while time < p.duration:
        n = len(alive)
        birth_rate = p.birth * (p.capacity - n)
        collapse_rate = p.collapse * n
        search_rate = p.search * n
        total_rate = birth_rate + collapse_rate + search_rate
        if total_rate == 0:
            if n == 0:
                empty_time += p.duration - time
            break
        next_time = time + rng.expovariate(total_rate)
        if n == 0:
            empty_time += min(next_time, p.duration) - time
        if next_time > p.duration:
            break
        time = next_time
        draw = rng.random() * total_rate
        source = target = source_id = target_id = -1
        if draw < birth_rate:
            empty_sites = [site for site in range(p.capacity) if site not in alive]
            target = rng.choice(empty_sites)
            target_id = next_id
            next_id += 1
            alive[target] = target_id
            kind = "birth"
            counts[kind] += 1
        elif draw < birth_rate + collapse_rate:
            source = rng.choice(sorted(alive))
            source_id = alive.pop(source)
            kind = "collapse"
            counts[kind] += 1
        else:
            source = rng.choice(sorted(alive))
            source_id = alive[source]
            target = rng.randrange(p.capacity - 1)
            if target >= source:
                target += 1
            target_id = alive.get(target, -1)
            counts["search"] += 1
            if target_id < 0:
                kind = "empty_search"
            elif rng.random() < p.discovery:
                kind = "attack"
                alive.pop(target)
                counts[kind] += 1
            else:
                kind = "undetected_search"
        events.append(Event(time, kind, source, target, source_id, target_id,
                            n, len(alive)))

    changes = [event for event in events if event.count_before != event.count_after]
    change_times = [event.time for event in changes]
    change_counts = [p.initial] + [event.count_after for event in changes]
    steps = int(round(p.duration / p.sample_interval))
    times = [index * p.sample_interval for index in range(steps + 1)]
    if times[-1] < p.duration - 1e-10:
        times.append(p.duration)
    times = [min(t, p.duration) for t in times if t <= p.duration + 1e-10]
    samples = [change_counts[bisect_right(change_times, t)] for t in times]
    return Trial(p, seed, initial_sites, events, times, samples, len(alive),
                 counts["birth"], counts["collapse"], counts["attack"],
                 counts["search"], empty_time)


def rhs(x, p: Parameters):
    return p.birth * (1 - x) - p.collapse * x - p.search * p.discovery * x * x


def equilibrium(p: Parameters):
    """Stable physical equilibrium, evaluated without cancellation near p=0."""
    a = p.search * p.discovery
    c = p.birth + p.collapse
    if p.birth == 0:
        return 0.0
    return 2 * p.birth / (c + math.sqrt(c * c + 4 * a * p.birth))


def ode_solution(time, x0, p: Parameters):
    """Closed-form solution of x'=b-(b+mu)x-a*x^2, b>0."""
    a = p.search * p.discovery
    c = p.birth + p.collapse
    if a == 0:
        if c == 0:
            return x0
        xstar = p.birth / c
        return xstar + (x0 - xstar) * math.exp(-c * time)
    discriminant = math.sqrt(c * c + 4 * a * p.birth)
    positive_root = equilibrium(p)
    negative_root = -(c + discriminant) / (2 * a)
    ratio = ((x0 - positive_root) / (x0 - negative_root)
             * math.exp(-discriminant * time))
    return (positive_root - ratio * negative_root) / (1 - ratio)


def stationary_distribution(p: Parameters):
    """Exact stationary count law of the finite, irreducible birth-death chain.

    lambda_n=b(M-n); delta_n=mu*n+s*p*n*(n-1)/(M-1).
    Every unsuccessful search is a state-preserving event and drops out here.
    """
    if p.birth <= 0 or p.collapse <= 0:
        raise ValueError("This stationary formula assumes b>0 and mu>0.")
    logs = [0.0]
    for n in range(1, p.capacity + 1):
        birth = p.birth * (p.capacity - n + 1)
        death = p.collapse * n + p.search * p.discovery * n * (n - 1) / (p.capacity - 1)
        logs.append(logs[-1] + math.log(birth / death))
    peak = max(logs)
    weights = [math.exp(value - peak) for value in logs]
    total = sum(weights)
    return [value / total for value in weights]


def parameters_from_config(config, discovery):
    return Parameters(side=config["side"], initial=config["initial_civilizations"],
                      birth=config["birth_rate_per_empty_star"],
                      collapse=config["collapse_rate_per_civilization"],
                      search=config["search_rate_per_civilization"],
                      discovery=discovery, duration=config["duration"],
                      sample_interval=config["sample_interval"])
