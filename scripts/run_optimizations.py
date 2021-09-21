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

# kwargs for extracellular computation
extra_kwargs = dict(fs=20,
                    fcut=[300, 6000],
                    filt_type="filtfilt",
                    ms_cut=[3, 10])


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Run optimization for multi-modal fitting.",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="The seed used for the optimization")
    parser.add_argument("--cell-folder", type=str, default="../cell_models",
                        help="The folder where the cell models are (default '../cell_models/')")
    parser.add_argument("--feature-set", type=str, default="extra",
                        help="The feature set to be used ('soma' - 'extra')")
    parser.add_argument("--model", type=str, default="hay",
                        help="the model to be optimized ('hay' - 'hay_ais' - 'hay_ais_hillock')")
    parser.add_argument("--extra-strategy", type=str, default="all",
                        help="The strategy for using extracellular features ('all' - 'single' - 'sections')")
    parser.add_argument("--ipyparallel", action="store_true", default=False,
                        help="If True ipyparallel is used to parallelize computations (default False)")
    parser.add_argument("--data-folder", type=str, default=None,
                        help="The folder containing the features and protocol json files")
    parser.add_argument("--opt-folder", type=str, default=None,
                        help="The folder containing the results of optimization "
                             "(default is parent of data_folder/optimization_results)")
    parser.add_argument("--offspring", type=int, default=20,
                        help="The population size (offspring) - default 20")
    parser.add_argument("--maxgen", type=int, default=2000,
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


def get_cp_filename(opt_folder, model, feature_set, extra_strategy, seed):

    if extra_strategy is not None:
        cp_filename = opt_folder / 'checkpoints' / \
                      f'model={model}_featureset={feature_set}_strategy={extra_strategy}_seed={seed}'
    else:
        cp_filename = opt_folder / 'checkpoints' / f'model={model}_featureset={feature_set}_seed={seed}'

    if not cp_filename.parent.is_dir():
        os.makedirs(cp_filename.parent)

    return cp_filename


def main():

    args = get_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        stream=sys.stdout,
    )

    map_function = get_mapper(args)

    protocols_with_lfp = None
    timeout = 300.
    if args.feature_set == "extra":
        protocols_with_lfp = ['IDrest_300']
        timeout = 900.

    probe_file = None

    if args.data_folder is not None:
        # Load features / protocols / and probe
        data_folder = Path(args.data_folder)
        
        if args.extra_strategy:
            feature_file = data_folder / f"features_BPO_{args.extra_strategy}.json"
            protocol_file = data_folder / f"protocols_BPO_{args.extra_strategy}.json"
        else:
            feature_file = data_folder / "features_BPO.json"
            protocol_file = data_folder / "protocols_BPO.json"

        if not Path(feature_file).is_file():
            raise Exception("Couldn't find a feature json file in the provided folder.")
        if not Path(protocol_file).is_file():
            raise Exception("Couldn't find a protocol json file in the provided folder.")

        if args.feature_set == "extra":
            probe_file = data_folder / f"probe_BPO.json"
            if not os.path.isfile(probe_file):
                raise Exception("Couldn't find a probe json file in the provided folder.")
    else:
        raise Exception("Provide --folder argument to specify where BPO files are")

    model_name = args.model

    assert args.cell_folder is not None, "Provide --cell-folder argument to specify where cell models folders are"
    cell_folder = Path(args.cell_folder) / f"{model_name}_model"
    
    eva = mf.create_evaluator(
        model_name=model_name,
        cell_model_folder=cell_folder,
        feature_set=args.feature_set,
        feature_file=feature_file,
        protocol_file=protocol_file,
        probe_file=probe_file,
        protocols_with_lfp=protocols_with_lfp,
        extra_recordings=None,
        timeout=timeout,
        **extra_kwargs
    )
    
    opt = bluepyopt.deapext.optimisationsCMA.DEAPOptimisationCMA(
        evaluator=eva,
        offspring_size=args.offspring,
        seed=args.seed,
        map_function=map_function,
        weight_hv=0.4,
        selector_name="multi_objective"
    )

    if args.opt_folder is None:
        opt_folder = data_folder.parent / "optimization_results"
    else:
        opt_folder = Path(args.opt_folder)

    cp_filename = get_cp_filename(opt_folder, args.model, args.feature_set, args.extra_strategy, args.seed)

    if cp_filename.is_file():
        logger.info(f"Continuing from checkpoint: {cp_filename}")
        continue_cp = True
    else:
        logger.info(f"Saving checkpoint in: {cp_filename}")
        continue_cp = False

    # save evaluator args
    eva_args = dict(model_name=model_name,
                    cell_model_folder=str(cell_folder),
                    feature_set=args.feature_set,
                    feature_file=str(feature_file),
                    protocol_file=str(protocol_file),
                    probe_file=str(probe_file),
                    protocols_with_lfp=protocols_with_lfp,
                    extra_recordings=None,
                    timeout=timeout)
    eva_args.update(extra_kwargs)

    eva_file = cp_filename.parent / f"{cp_filename.stem}.json"
    print(eva_file)
    with eva_file.open("w") as f:
        json.dump(eva_args, f, indent=4)

    opt.run(max_ngen=args.maxgen, cp_filename=cp_filename, continue_cp=continue_cp)


if __name__ == '__main__':
    main()
