import argparse
import logging
from multiprocessing import pool

import os
import sys
import json
from pathlib import Path
from datetime import datetime

import bluepyopt
import multiprocessing


import multimodalfitting as mf
from multimodalfitting.utils import _extra_kwargs
logger = logging.getLogger()

logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
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
    parser.add_argument("--strategy", type=str, default="soma",
                        help="The feature set to be used ('soma' - 'all' - 'single' - 'extra')")
    parser.add_argument("--model", type=str, default="hay",
                        help="the model to be optimized ('hay' - 'hay_ais' - 'hay_ais_hillock' or experimental ones)")
    parser.add_argument("--sim", type=str, default="lfpy",
                        help="the simulator to be used ('lfpy' - 'neuron')")
    parser.add_argument("--ipyparallel", action="store_true", default=False,
                        help="If True ipyparallel is used to parallelize computations (default False)")
    parser.add_argument("--multiprocessing", action="store_true", default=False,
                        help="If True multiprocessing is used to parallelize computations (default False)")
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
                        help="The maximum number of generations - default 600")
    parser.add_argument("--timeout", type=int, default=900,
                        help="Maximum run time for a protocol (in seconds)")
    parser.add_argument("--cm_ra", type=int, default=0,
                        help="If 1 and model is 'experimental' and cm (seperately) and Ra (global) is is optimized")

    return parser


class NoDaemonProcess(multiprocessing.Process):
    """Class that represents a non-daemon process"""

    # pylint: disable=dangerous-default-value

    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}):
        """Ensures group=None, for macosx."""
        super().__init__(group=None, target=target, name=name, args=args, kwargs=kwargs)

    def _get_daemon(self):
        """Get daemon flag"""
        return False

    def _set_daemon(self, value):
        """Set daemon flag"""

    daemon = property(_get_daemon, _set_daemon)


class NestedPool(pool.Pool):  # pylint: disable=abstract-method
    """Class that represents a MultiProcessing nested pool"""

    Process = NoDaemonProcess


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
    
    elif args.multiprocessing:

        nested_pool = NestedPool()
        return nested_pool.map

    return None


def get_cp_filename(opt_folder, model, strategy, seed, abd, ra, cm_ra):

    cp_name = f'model={model}_strategy={strategy}'

    if abd:
        cp_name = cp_name + "_abd"
    if ra:
        cp_name = cp_name + "_ra"
    if cm_ra:
        cp_name = cp_name + "_cm_ra"

    cp_name = cp_name + f"_seed={seed}"

    cp_filename = opt_folder / 'checkpoints' / cp_name

    if not cp_filename.parent.is_dir():
        os.makedirs(cp_filename.parent)

    return cp_filename


def save_evaluator_configuration(
    model_name,
    strategy,
    protocols_with_lfp,
    cell_folder,
    timeout,
    cp_filename,
    simulator,
    abd,
    optimize_ra,
    cm_ra
):

    eva_args = dict(model_name=model_name,
                    strategy=strategy,
                    protocols_with_lfp=protocols_with_lfp,
                    cell_folder=str(cell_folder),
                    extra_recordings=None,
                    timeout=timeout,
                    simulator=simulator,
                    abd=abd,
                    cm_ra=cm_ra,
                    optimize_ra=optimize_ra)
    eva_args.update(_extra_kwargs)

    eva_file = cp_filename.parent / f"{cp_filename.stem}.json"

    with eva_file.open("w") as f:
        json.dump(eva_args, f, indent=4)


def main():

    args = get_parser().parse_args()

    if args.opt_folder is None:
        opt_folder = Path(".") / "optimization_results" 
    else:
        opt_folder = Path(args.opt_folder)

    map_function = get_mapper(args)

    if args.strategy in ["all", "sections", "single"] and args.sim == "neuron":
        raise Exception(
            f"With strategy {args.strategy}, please use the lfpy simulator.")

    protocols_with_lfp = ['IDrest_300'] if args.strategy in ["all", "sections", "single"] else None

    eva = mf.create_evaluator(
        model_name=args.model,
        strategy=args.strategy,
        protocols_with_lfp=protocols_with_lfp,
        cell_folder=Path(args.cell_folder),
        extra_recordings=None,
        timeout=args.timeout,
        simulator=args.sim,
        abd=args.abd,
        optimize_ra=args.ra,
        cm_ra=args.cm_ra,
        **_extra_kwargs
    )

    opt = bluepyopt.deapext.optimisationsCMA.DEAPOptimisationCMA(
        evaluator=eva,
        offspring_size=args.offspring,
        seed=args.seed,
        map_function=map_function,
        weight_hv=0.4,
        selector_name="multi_objective"
    )

    cp_filename = get_cp_filename(
        opt_folder, args.model, args.strategy, args.seed, args.abd, args.ra, args.cm_ra
    )

    if cp_filename.is_file():
        logger.info(f"Continuing from checkpoint: {cp_filename}")
        continue_cp = True
    else:
        logger.info(f"Saving checkpoint in: {cp_filename}")
        continue_cp = False

    save_evaluator_configuration(
        model_name=args.model,
        strategy=args.strategy,
        protocols_with_lfp=protocols_with_lfp,
        cell_folder=Path(args.cell_folder),
        timeout=args.timeout,
        cp_filename=cp_filename,
        simulator=args.sim,
        abd=args.abd,
        optimize_ra=args.ra,
        cm_ra=args.cm_ra
    )

    opt.run(max_ngen=args.maxgen, cp_filename=str(cp_filename), continue_cp=continue_cp)


if __name__ == '__main__':
    main()
