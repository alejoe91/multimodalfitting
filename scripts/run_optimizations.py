import argparse
import logging
import os
import sys
from pathlib import Path
import json
from datetime import datetime

import bluepyopt

import multimodalfitting as mf

logger = logging.getLogger()

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
)

# kwargs for extracellular computation
EXTRA_EVALUATOR_KWARGS = dict(
    fs=20,
    fcut=[300, 6000],
    filt_type="filtfilt",
    ms_cut=[3, 10]
)


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Run optimization for multi-modal fitting.",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="The seed used for the optimization")
    parser.add_argument("--cell-folder", type=str, default=None, required=False,
                        help="The folder where the cell models are (default 'multimodalfitting/cell_models/')")
    parser.add_argument("--feature-set", type=str, default="extra",
                        help="The feature set to be used ('soma' - 'extra')")
    parser.add_argument("--model", type=str, default="hay",
                        help="the model to be optimized ('hay' - 'hay_ais' - 'hay_ais_hillock' or experimental ones)")
    parser.add_argument("--sim", type=str, default="lfpy",
                        help="the simulator to be used ('lfpy' - 'neuron')")
    parser.add_argument("--extra-strategy", type=str, default="all",
                        help="The strategy for using extracellular features ('all' - 'single' - 'sections')")
    parser.add_argument("--ipyparallel", action="store_true", default=False,
                        help="If True ipyparallel is used to parallelize computations (default False)")
    parser.add_argument("--opt-folder", type=str, default=None, required=True,
                        help="The folder containing the results of optimization "
                             "(default is parent of ./optimization_results)")
    parser.add_argument("--abd", type=int, default=0,
                        help="If True and model is 'experimental', the ABD section is used")
    parser.add_argument("--ra", type=int, default=0,
                        help="If True and model is 'experimental' and abd is used, Ra in ABD and AIS is also optimized")
    parser.add_argument("--offspring", type=int, default=20,
                        help="The population size (offspring) - default 20")
    parser.add_argument("--maxgen", type=int, default=600,
                        help="The maximum number of generations - default 2000")

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


def get_cp_filename(opt_folder, model, feature_set, extra_strategy, seed, abd, ra):

    cp_folder = opt_folder / 'checkpoints'
    if extra_strategy is not None:
        cp_name = f'model={model}_featureset={feature_set}_strategy={extra_strategy}'
    else:
        cp_name = f'model={model}_featureset={feature_set}'

    if abd:
        cp_name = cp_name + "_abd"

    if ra:
        cp_name = cp_name + "_ra"

    # add seed
    cp_name = cp_name + f"_seed={seed}"

    cp_filename = cp_folder / cp_name

    if not cp_filename.parent.is_dir():
        os.makedirs(cp_filename.parent)

    return cp_filename


def save_evaluator_configuration(
    model_name,
    feature_set,
    extra_strategy,
    protocols_with_lfp,
    cell_folder,
    timeout,
    cp_filename,
    simulator,
    abd,
    optimize_ra
):

    eva_args = dict(model_name=model_name,
                    feature_set=feature_set,
                    extra_strategy=extra_strategy,
                    protocols_with_lfp=protocols_with_lfp,
                    cell_folder=str(cell_folder),
                    extra_recordings=None,
                    timeout=timeout,
                    simulator=simulator,
                    abd=abd,
                    optimize_ra=optimize_ra)

    eva_args.update(EXTRA_EVALUATOR_KWARGS)

    eva_file = cp_filename.parent / f"{cp_filename.stem}.json"
    print(eva_file)
    with eva_file.open("w") as f:
        json.dump(eva_args, f, indent=4)


def main():

    args = get_parser().parse_args()

    if args.opt_folder is None:
        opt_folder = Path(".") / "optimization_results" 
    else:
        opt_folder = Path(args.opt_folder)

    map_function = get_mapper(args)

    sim = args.sim
    feature_set = args.feature_set

    if feature_set == "extra" and sim == "neuron":
        print("For 'extra' features use the lfpy simulator. Setting feature_set to 'soma'")
        feature_set = "soma"

    protocols_with_lfp = None
    timeout = 300.
    if feature_set == "extra":
        protocols_with_lfp = ['IDrest_300']
        timeout = 900.

    eva = mf.create_evaluator(
        model_name=args.model,
        feature_set=feature_set,
        extra_strategy=args.extra_strategy,
        protocols_with_lfp=protocols_with_lfp,
        cell_folder=Path(args.cell_folder),
        extra_recordings=None,
        timeout=timeout,
        simulator=sim,
        abd=args.abd,
        optimize_ra=args.ra,
        **EXTRA_EVALUATOR_KWARGS
    )

    opt = bluepyopt.deapext.optimisationsCMA.DEAPOptimisationCMA(
        evaluator=eva,
        offspring_size=args.offspring,
        seed=args.seed,
        map_function=map_function,
        weight_hv=0.4,
        selector_name="multi_objective"
    )

    # add abd and ra
    cp_filename = get_cp_filename(
        opt_folder, args.model, args.feature_set, args.extra_strategy, args.seed, args.abd, args.ra
    )

    if cp_filename.is_file():
        logger.info(f"Continuing from checkpoint: {cp_filename}")
        continue_cp = True
    else:
        logger.info(f"Saving checkpoint in: {cp_filename}")
        continue_cp = False

    save_evaluator_configuration(
        model_name=args.model,
        feature_set=args.feature_set,
        extra_strategy=args.extra_strategy,
        protocols_with_lfp=protocols_with_lfp,
        cell_folder=Path(args.cell_folder),
        timeout=timeout,
        cp_filename=cp_filename,
        simulator=sim,
        abd=args.abd,
        optimize_ra=args.ra
    )

    opt.run(max_ngen=args.maxgen, cp_filename=str(cp_filename), continue_cp=continue_cp)


if __name__ == '__main__':
    main()
