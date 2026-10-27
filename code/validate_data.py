"""Reconstruct every saved realization and audit the submitted experiment."""
import csv
import gzip
import json
from pathlib import Path
from collections import Counter
from bisect import bisect_right

ROOT=Path(__file__).resolve().parents[1]


def main():
    config=json.loads((ROOT/'config/experiment.json').read_text())
    runs=list(csv.DictReader((ROOT/'data/runs.csv').open()))
    parameters=Counter(float(run['p']) for run in runs)
    assert len(runs)==220 and set(parameters.values())=={20}
    assert len({int(run['seed']) for run in runs})==220
    samples={}
    with (ROOT/'data/trajectories.csv').open() as stream:
        for row in csv.DictReader(stream):
            samples.setdefault((float(row['p']),int(row['repetition'])),[]).append((float(row['time']),int(row['count'])))
    total_events=0
    for run in runs:
        p=float(run['p'])
        repetition=int(run['repetition'])
        # The simulator's seed reproduces the initial locations, independently
        # of any subsequently logged changes.
        import random
        initial=sorted(random.Random(int(run['seed'])).sample(range(config['side']**2),int(run['initial_count'])))
        alive={site:index for index,site in enumerate(initial)}
        known_ids=set(alive.values())
        n=int(run['initial_count'])
        previous=0
        counts=Counter()
        changes=[]
        with gzip.open(ROOT/'data'/run['event_file'],'rt') as stream:
            for event in csv.DictReader(stream):
                total_events+=1
                time=float(event['time'])
                assert previous<time<=config['duration']
                previous=time
                source,target=int(event['source']),int(event['target'])
                sid,tid=int(event['source_id']),int(event['target_id'])
                assert int(event['count_before'])==len(alive)==n
                kind=event['kind']
                if kind=='birth':
                    assert target not in alive and tid not in known_ids
                    alive[target]=tid
                    known_ids.add(tid)
                    counts['birth']+=1
                elif kind=='collapse':
                    assert alive[source]==sid
                    del alive[source]
                    counts['collapse']+=1
                else:
                    counts['search']+=1
                    assert source!=target and alive[source]==sid
                    if kind=='attack':
                        assert p>0 and alive[target]==tid
                        del alive[target]
                        counts['attack']+=1
                    elif kind=='empty_search':
                        assert target not in alive and tid==-1
                    else:
                        assert kind=='undetected_search' and p<1 and alive[target]==tid
                after=len(alive)
                assert after==int(event['count_after'])
                if after!=n: changes.append((time,after))
                n=after
        assert n==int(run['final_count'])
        assert n==int(run['initial_count'])+counts['birth']-counts['collapse']-counts['attack']
        for key,column in [('birth','births'),('collapse','collapses'),('attack','attacks'),('search','searches')]:
            assert counts[key]==int(run[column])
        change_times=[time for time,n in changes]
        change_counts=[int(run['initial_count'])]+[n for time,n in changes]
        for time,value in samples[(p,repetition)]:
            assert value==change_counts[bisect_right(change_times,time)]
    report={"status":"passed","runs":len(runs),"events_reconstructed":total_events,
            "unique_seeds":220,"repetitions_per_probability":20,
            "ever_empty_runs":sum(int(row['ever_empty']) for row in runs),
            "empty_at_T_runs":sum(int(row['empty_at_T']) for row in runs),
            "checks":["event chronology","unique civilization IDs","no self-attacks",
                      "valid occupancy transitions","population accounting",
                      "all sampled trajectories match event logs","220 unique seeds"]}
    (ROOT/'data/validation.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
