"""Hand-counted metric examples, no real data or training."""
import json
from pathlib import Path
import numpy as np
from metric_supplement import alignment_metrics

Y = np.array([[1,0], [0,1], [1,0], [0,1]])
P = np.array([[.9,.8], [.8,.7], [.7,.6], [.1,.1]])
m = alignment_metrics(Y, P, np.array([.5,.5]), ["a","b"])
assert m["per_label"]["a"]["tp"] == 2
assert m["per_label"]["a"]["tn"] == 1
assert m["per_label"]["a"]["fp"] == 1
assert m["per_label"]["a"]["fn"] == 0
assert m["accuracy_macro_binary"] == .5
assert m["subset_accuracy"] == 0
assert m["specificity_macro"] == .25
assert np.isclose(m["per_label"]["a"]["average_precision"], 5/6)
assert np.isclose(m["per_label"]["a"]["auprc_trapezoid"], 19/24)
assert np.isclose(m["accuracy_macro_binary"], 1-m["hamming_loss"])
undefined = alignment_metrics(np.array([[1,0],[1,0]]),
    np.array([[.8,.2],[.7,.3]]), np.array([.5,.5]), ["all_positive","all_negative"])
assert undefined["per_label"]["all_positive"]["specificity"] is None
assert undefined["per_label"]["all_negative"]["auprc_trapezoid"] is None
assert undefined["specificity_valid_labels"] == 1
assert undefined["auprc_trapezoid_valid_labels"] == 1
json.dumps(undefined, allow_nan=False)
invalid_cases = [
    (np.array([[2,0]]), np.array([[.5,.5]]), [.5,.5], ["a","b"]),
    (Y, P * 2, [.5,.5], ["a","b"]),
    (Y, P, [.5,.5], ["a","a"]),
]
for arguments in invalid_cases:
    try:
        alignment_metrics(*arguments)
    except ValueError:
        continue
    raise AssertionError("Invalid metric input was accepted")
result={"status":"PASS","checks":["hand-counted confusion matrix",
    "binary accuracy versus subset accuracy","AP versus trapezoidal PR area",
    "undefined denominator handling","strict JSON","invalid inputs"],
    "training_started":False,"real_test_predictions_loaded":False}
Path(__file__).with_name("metric_validation.json").write_text(
    json.dumps(result,indent=2)+"\n",encoding="utf-8")
print(json.dumps(result))
