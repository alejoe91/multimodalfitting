import argparse
import logging
import os
import sys
import textwrap
from datetime import datetime

import pandas as pd

import evaluator

logger = logging.getLogger()


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="LFPy",
        epilog=textwrap.dedent(""),
    )
    parser.add_argument(
        "--offspring_size",
        type=int,
        required=False,
        default=250,
        help="number of individuals in offspring",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature_set", type=str, default="extra")
    parser.add_argument("--sample_id", type=int, required=True)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        dest="verbose",
        default=0,
        help="-v for INFO, -vv for DEBUG",
    )
    parser.add_argument(
        "--ipyparallel", action="store_true", default=False, help="Use ipyparallel"
    )
    return parser


def get_mapper(args):
    if args.ipyparallel or os.getenv("USEIPYP"):
        from ipyparallel import Client

        rc = Client(profile=os.getenv("IPYTHON_PROFILE"))

        logger.debug("Using ipyparallel with %d engines", len(rc))

        lview = rc.load_balanced_view()

        def mapper(func, it):
            start_time = datetime.now()
            ret = lview.map_sync(func, it)
            logger.debug("Generation took %s", datetime.now() - start_time)
            return ret

        return mapper
    else:
        return None


args = get_parser().parse_args()

logging.basicConfig(
    level=(logging.WARNING, logging.INFO, logging.DEBUG)[args.verbose],
    stream=sys.stdout,
)

#map_function = get_mapper(args)
map_function = None

channels = "map"
prob_type = "planar"

random_params_file = "config/params/smart_random.csv"
random_params = pd.read_csv(random_params_file, index_col="index")
params = random_params.iloc[0].to_dict()

prep = evaluator.prepare_optimization(
    feature_set=args.feature_set,
    sample_id=args.sample_id,
    offspring_size=args.offspring_size,
    channels=channels,
    probe_type=prob_type,
    map_function=map_function,
    seed=args.seed,
)

opt = prep["optimisation"]
eva = prep["evaluator"]
fitness_calculator = prep["objectives_calculator"]
fitness_protocols = prep["protocols"]

out = evaluator.run_optimization(
    feature_set=args.feature_set,
    sample_id=args.sample_id,
    opt=opt,
    max_ngen=20000,
    channels=channels,
    seed=args.seed,
    prob_type=prob_type,
)
