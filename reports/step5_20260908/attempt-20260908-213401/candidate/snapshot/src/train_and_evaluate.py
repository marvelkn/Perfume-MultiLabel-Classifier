"""Explicit stages: preparation never starts training; test evaluation is separate."""
import argparse
from .experiments import initialize, tune, baseline, finalize, evaluate_test
from .runtime import ResourceLimit
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command",required=True)
    init = sub.add_parser("init")
    init.add_argument("--dataset",required=True)
    init.add_argument("--run",required=True)
    for command in ("baseline","tune","fit","evaluate"):
        p = sub.add_parser(command)
        p.add_argument("--run",required=True)
        p.add_argument("--model",choices=["xgb","lgbm"],required=True)
        if command != "evaluate":
            p.add_argument("--temperature-file")
        if command == "tune":
            p.add_argument("--trials",type=int,required=True)
            p.add_argument("--include-mlsmote",action="store_true",help="Enable the explicitly experimental synthetic-feature candidate")
    args = parser.parse_args()
    try:
        if args.command == "init":
            print(initialize(args.dataset,args.run))
        elif args.command == "tune":
            strategies = ("none","class_weight","random_oversample") + (("mlsmote",) if args.include_mlsmote else ())
            tune(args.run,args.model,args.trials,args.temperature_file,strategies)
        elif args.command == "baseline":
            print(baseline(args.run,args.model,args.temperature_file))
        elif args.command == "fit":
            print(finalize(args.run,args.model,args.temperature_file))
        else:
            print(evaluate_test(args.run,args.model))
    except ResourceLimit as exc:
        parser.exit(2,f"Training stopped: {exc}\nCompleted trials remain saved.\n")
if __name__ == "__main__":
    main()
