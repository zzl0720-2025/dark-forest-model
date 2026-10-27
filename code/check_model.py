"""Checks of event semantics, reproducibility, and known mathematical limits."""
from dataclasses import replace
import math
import statistics
import unittest
from model import Parameters, simulate, rhs, equilibrium, ode_solution, stationary_distribution


class ModelChecks(unittest.TestCase):
    def test_events_and_replay_are_consistent(self):
        p = Parameters(side=5, initial=15, birth=.03, collapse=.04,
                       search=2, discovery=.8, duration=60)
        trial = simulate(p, 240)
        alive = set(trial.initial_sites)
        last_time = 0
        for event in trial.events:
            self.assertGreater(event.time, last_time)
            last_time = event.time
            self.assertEqual(len(alive), event.count_before)
            if event.kind == "birth":
                self.assertNotIn(event.target, alive)
                alive.add(event.target)
            elif event.kind == "collapse":
                self.assertIn(event.source, alive)
                alive.remove(event.source)
            else:
                self.assertIn(event.source, alive)
                self.assertNotEqual(event.source, event.target)
                if event.kind == "attack":
                    self.assertIn(event.target, alive)
                    alive.remove(event.target)
                elif event.kind == "empty_search":
                    self.assertNotIn(event.target, alive)
                else:
                    self.assertIn(event.target, alive)
            self.assertEqual(len(alive), event.count_after)
            self.assertLessEqual(len(alive), p.capacity)
        self.assertEqual(len(alive), trial.final_count)
        self.assertEqual(trial.final_count, p.initial + trial.births - trial.collapses - trial.attacks)
        self.assertEqual(trial.events, simulate(p, 240).events)

    def test_zero_discovery_and_last_civilization(self):
        p = Parameters(side=4, initial=8, discovery=0, duration=30)
        self.assertEqual(simulate(p, 5).attacks, 0)
        last = replace(p, initial=1, birth=0, collapse=0, discovery=1)
        self.assertEqual(simulate(last, 9).final_count, 1)
        empty = replace(last, initial=0)
        self.assertEqual(simulate(empty, 9).final_count, 0)

    def test_analytic_solution_and_stability(self):
        for prob in (0, .01, .5, 1):
            p = Parameters(discovery=prob)
            star = equilibrium(p)
            self.assertAlmostEqual(rhs(star, p), 0, places=13)
            for x0 in (0, .2, 1):
                self.assertAlmostEqual(ode_solution(0, x0, p), x0)
                for t in (.1, 10, 100):
                    h = 1e-5
                    x = ode_solution(t, x0, p)
                    numerical = (ode_solution(t+h, x0, p) - ode_solution(t-h, x0, p))/(2*h)
                    self.assertAlmostEqual(numerical, rhs(x, p), places=8)
                    self.assertTrue(0 <= x <= 1)

    def test_stationary_detailed_balance(self):
        for prob in (0, .5, 1):
            p = Parameters(discovery=prob)
            law = stationary_distribution(p)
            self.assertAlmostEqual(sum(law), 1)
            for n in range(p.capacity):
                up = law[n]*p.birth*(p.capacity-n)
                down = law[n+1]*(p.collapse*(n+1)+p.search*prob*(n+1)*n/(p.capacity-1))
                self.assertAlmostEqual(up, down, places=13)
            if prob == 0:
                q = p.birth/(p.birth+p.collapse)
                self.assertAlmostEqual(sum(n*v for n,v in enumerate(law)), p.capacity*q)

    def test_no_attack_ensemble_matches_known_mean(self):
        p = Parameters(side=5, initial=20, birth=.02, collapse=.03,
                       search=0, discovery=0, duration=10)
        outcomes = [simulate(p, seed).final_count for seed in range(500)]
        exact = p.capacity*ode_solution(p.duration,p.initial/p.capacity,p)
        se = statistics.stdev(outcomes)/math.sqrt(len(outcomes))
        self.assertLess(abs(statistics.mean(outcomes)-exact), 4*se)


if __name__ == "__main__":
    unittest.main(verbosity=2)
