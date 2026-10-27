DARK FOREST SIMULATION CODE

Model
-----
There are 100 fixed stars. Each is empty or occupied by one civilization.
All civilizations follow the same rules: independent emergence at empty stars,
internal collapse, and destruction of another civilization after discovering it.
Search is global, targets exclude the searcher, and events occur one at a time
in continuous time. Positions do not affect search probability.

The main experiment varies only discovery probability p, using 0, 0.1, ..., 1.
Every p has 20 independent runs, each beginning with 20 civilizations and ending
at T=500. Birth, collapse, and search rates are fixed at 0.001, 0.01, and 1.
The model uses arbitrary time units.


Run the standalone code archive on computer
--------------------------------------------------
Unzip the code archive, open a terminal in its top-level folder, and run:

    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python code/check_model.py
    .venv/bin/python code/run_all.py
    .venv/bin/python code/validate_data.py

On Windows use .venv\Scripts\python.exe in place of .venv/bin/python.
Use Python 3.11 or later. requirements-lock.txt records the package versions
used for the saved experiment. No report files or TeX installation are needed.

The first command should finish with OK. The experiment prints the mean final
population at each p. The final audit should report status "passed" and 220 runs.
Rerunning uses the saved seed rule and replaces the generated experiment files.

Files
-----
code/model.py                  Agent simulation and analytical formulas.
code/run_all.py                Experiment, statistics, figures, and replay.
code/check_model.py            Mathematical and event-rule checks.
code/validate_data.py          Reconstructs and checks every saved realization.
code/viewer_template.html      Template required to regenerate the replay.
code/numbered/                 Line-numbered source copies.
config/experiment.json         Experimental settings.
data/runs.csv                 One row per run, including its random seed.
data/summary.csv              Means, SDs, confidence intervals, and references.
data/trajectories.csv         Counts recorded every 0.5 model time units.
data/events/*.csv.gz          Complete event records, including failed searches.
data/stationary_distributions.csv  Exact finite-count stationary probabilities.
data/replay.json              Saved records for the three displayed examples.
data/manifest.json            Settings, software versions, and source hashes.
data/validation.json          Event-record audit results.
figures/                     Generated figures in PDF, SVG, and PNG formats.
visualization/index.html      Offline event replay. Open this in a browser.

Reading the outputs
-------------------
The primary result is N(T), the final number of civilizations. Each confidence
interval estimates the mean of 20 independent runs; it does not describe the
range of individual outcomes. The supplied interval formula uses 19 degrees
of freedom and assumes the assignment's 20 repetitions.

The replay shows the first repetition at p=0, 0.5, and 1. Playback controls
change how saved events are displayed; they do not recalculate the model.
The template is included so that run_all.py can rebuild the replay.

For the supplied settings the mean final population is 9.10 at p=0 and 3.00
at p=1. Forty-four runs became empty temporarily, and none ended empty.
New civilizations can appear in an empty galaxy, so this is not permanent
extinction. The ODE has one stable physical equilibrium throughout p in [0,1].

The seed rule is 271026 + 10000*probability_index + repetition (1 through 20).
No seeds were replaced after inspecting the outcomes. After changing settings,
rerun the experiment and update any written interpretation of its results.
