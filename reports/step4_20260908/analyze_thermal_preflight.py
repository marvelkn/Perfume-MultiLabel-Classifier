"""Summarize the recorded Step 4 thermal preflight without loading ML data."""
import hashlib
import json
import statistics
from datetime import datetime
from itertools import pairwise
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
HISTORY = DIRECTORY / 'pre-smoke-idle-01.jsonl'

def main():
    rows = [json.loads(line) for line in HISTORY.read_text(encoding='utf-8').splitlines() if line]
    summary = json.loads(HISTORY.with_suffix('.summary.json').read_text(encoding='utf-8'))
    if not rows:
        raise ValueError('No actual sensor observations')
    by_timestamp = {}
    for row in rows:
        assert row['sensor_id'] == '/amdcpu/0/temperature/2'
        assert row['sensor_name'] == 'Core (Tctl/Tdie)'
        previous = by_timestamp.setdefault(row['timestamp'], row['cpu_c'])
        assert previous == row['cpu_c'], 'Conflicting values for a source timestamp'
    stamps = list(by_timestamp)
    source_gaps = [b-a for a,b in pairwise(stamps)]
    observer_gaps = [b['observed_timestamp']-a['observed_timestamp'] for a,b in pairwise(rows)]
    values = list(by_timestamp.values())
    last_300 = [r for r in rows if r['timestamp'] >= stamps[-1]-300]
    sensor_valid = (summary['status'] == 'RECORDED' and stamps[-1]-stamps[0] >= 600
                    and all(0 < gap <= 10 for gap in source_gaps)
                    and all(0 < gap <= 10 for gap in observer_gaps)
                    and all(-2 <= row['age_seconds'] <= 5 for row in rows))
    below_start = values[-1] < 75
    evidence = {
        'verified_at': datetime.now().astimezone().isoformat(),
        'recording_status': summary['status'],
        'sensor_recording_valid': sensor_valid,
        'sensor_identity_basis': 'CPU die identifier/name and fresh changing CSV match the sensor verified in Step 1; no new UI screenshot',
        'started_at': summary['started_at'], 'ended_at': summary['ended_at'],
        'recording_duration_seconds': summary['duration_seconds'],
        'source_span_seconds': stamps[-1]-stamps[0],
        'collector_observations': len(rows), 'unique_source_samples': len(stamps),
        'cpu_temperature_c': {'initial': values[0], 'minimum': min(values), 'maximum': max(values),
                              'mean_unique_samples': statistics.mean(values), 'median': statistics.median(values), 'end': values[-1]},
        'last_300_seconds_temperature_c': {'minimum': min(r['cpu_c'] for r in last_300), 'maximum': max(r['cpu_c'] for r in last_300)},
        'unique_samples_below_start_75c': sum(v < 75 for v in values),
        'max_source_gap_seconds': max(source_gaps, default=0),
        'max_observer_gap_seconds': max(observer_gaps, default=0),
        'max_observed_telemetry_age_seconds': max(r['age_seconds'] for r in rows),
        'system_cpu_percent': {'mean': statistics.mean(r['system_cpu_percent'] for r in rows[1:]),
                               'minimum': min(r['system_cpu_percent'] for r in rows[1:]),
                               'maximum': max(r['system_cpu_percent'] for r in rows[1:]),
                               'last_60_observations_mean': statistics.mean(r['system_cpu_percent'] for r in rows[-60:])},
        'peak_collector_rss_bytes': max(r['collector_rss_bytes'] for r in rows),
        'minimum_system_available_ram_bytes': min(r['system_available_ram_bytes'] for r in rows),
        'monitor_peak_ram_bytes': None,
        'memory_scope': 'Collector only; monitor/other applications not included in peak RSS',
        'preflight_start_temperature_condition_met_at_end': below_start,
        'step4_result': 'STOP' if not sensor_valid or not below_start else 'REQUIRES_FRESH_GUARD_AND_SMOKE',
        'probe_started': False, 'completed_label_fits': {'xgb': 0, 'lgbm': 0},
        'probe_active_seconds_used': 0, 'trial_runtime_estimates': None,
        'test_arrays_deserialized': False, 'dataset_loaded': False,
        'background_conditions': 'Chat, monitor, collector, and background services active; no ML fitting. Not a laboratory idle baseline. Ambient temperature and fan speed not measured.',
        'history_sha256': hashlib.sha256(HISTORY.read_bytes()).hexdigest(),
        'summary_sha256': hashlib.sha256(HISTORY.with_suffix('.summary.json').read_bytes()).hexdigest(),
    }
    (DIRECTORY/'thermal-validation.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(evidence,indent=2))

if __name__ == '__main__':
    main()