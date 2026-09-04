"""Read-only identity/split audit; no model fitting or predictive test scores."""
import argparse
from pathlib import Path
from .artifacts import write_json
from .experiments import load_dataset
from .splits import scaffold_groups

def audit(dataset):
    root=Path(dataset)
    X,Y,manifest=load_dataset(root)
    train=(root/"smiles_train.txt").read_text().splitlines()
    test=(root/"smiles_test.txt").read_text().splitlines()
    if len(train)!=len(Y) or len(test)!=manifest["n_test"]:raise ValueError("Identity counts differ")
    overlap=set(train)&set(test)
    if overlap or len(set(train))!=len(train) or len(set(test))!=len(test):raise ValueError("Duplicate molecular identity")
    train_groups=set(scaffold_groups(train));test_groups=scaffold_groups(test)
    return {"dataset_id":manifest["dataset_id"],"source_revision":manifest["source_revision"],
       "n_train":len(train),"n_test":len(test),"n_labels":Y.shape[1],"n_features":X.shape[1],
       "identity_overlap":len(overlap),"test_rows_with_scaffold_seen_in_train":sum(g in train_groups for g in test_groups),
       "scaffold_note":"Murcko scaffold, chirality retained; all acyclic molecules share the empty scaffold. This is a diagnostic, not measured generalization performance.",
       "train_rows_without_target_labels":int((Y.sum(axis=1)==0).sum()),"train_positives":dict(zip(manifest["labels"],map(int,Y.sum(axis=0)))),
       "source_audit":manifest["source_audit"],"label_missingness":manifest["label_missingness"],
       "evaluation_status":manifest["evaluation_status"],"feature_schema_id":manifest["feature_schema_id"]}

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--dataset",required=True);p.add_argument("--output",required=True)
    a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError(a.output)
    write_json(a.output,audit(a.dataset))
