import argparse

import logging
import os
import sys
import pathlib

from datetime import datetime

import bluepyopt
import evaluator

logger = logging.getLogger()


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="LFPy",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature_set", type=str, default="extra")
    parser.add_argument("--model", type=str, default="hay")
    parser.add_argument("--ipyparallel", action="store_true", default=False)
    parser.add_argument("--folder", type=str, default=None)
    parser.add_argument("--offspring", type=int, default=20)
    parser.add_argument("--maxgen", type=int, default=2000)

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


def get_cp_filename(model, feature_set, seed):

    cp_filename = pathlib.Path('optimization_results') / 'checkpoints' / \
        f'model={model}_featureset={feature_set}_seed={seed}'

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

    probe_file = None
    protocols_with_lfp = None
    timeout = 300.
    if args.feature_set == "extra":
        protocols_with_lfp = ['IDrest_300']
        timeout = 900.

    if args.folder is not None:
        # Load features / protocols / and probe
        folder = pathlib.Path(args.folder)

        feature_files = [p for p in folder.iterdir() if "features" in p.name and "BPO" not in p.name]
        feature_bpo_files = [p for p in folder.iterdir() if "features" in p.name and "BPO" in p.name]
        if len(feature_bpo_files) == 1:
            feature_file = feature_bpo_files[0]
        elif len(feature_files) == 1:
            feature_file = feature_files[0]
        else:
            raise Exception("Couldn't find a feature json file in the provided folder.")

        protocol_files = [p for p in folder.iterdir() if "protocols" in p.name and "BPO" not in p.name]
        protocol_bpo_files = [p for p in folder.iterdir() if "protocols" in p.name and "BPO" in p.name]
        if len(protocol_bpo_files) == 1:
            protocol_file = protocol_bpo_files[0]
        elif len(protocol_files) == 1:
            protocol_file = protocol_files[0]
        else:
            raise Exception("Couldn't find a protocol json file in the provided folder.")

        probe_files = [p for p in folder.iterdir() if "probe" in p.name and "BPO" not in p.name]
        probe_bpo_files = [p for p in folder.iterdir() if "probe" in p.name and "BPO" in p.name]
        if len(probe_files) == 1:
            probe_file = probe_files[0]
        elif len(probe_bpo_files) == 1:
            probe_file = probe_bpo_files[0]
        elif args.feature_set == "extra":
            raise Exception("Couldn't find a probe json file in the provided folder.")

    eva = evaluator.create_evaluator(
        model_name=args.model,
        feature_set=args.feature_set,
        feature_file=feature_file,
        protocol_file=protocol_file,
        probe_file=probe_file,
        protocols_with_lfp=protocols_with_lfp,
        extra_recordings=None,
        timeout=timeout,
        fs=20,
        fcut=300,
        ms_cut=[2, 10],
        upsample=10
    )
    
    opt = bluepyopt.deapext.optimisationsCMA.DEAPOptimisationCMA(
        evaluator=eva,
        offspring_size=args.offspring,
        seed=args.seed,
        map_function=map_function,
        weight_hv=0.4,
        selector_name="multi_objective"
    )

    cp_filename = get_cp_filename(args.model, args.feature_set, args.seed)

    if cp_filename.is_file():
        logger.info(f"Continuing from checkpoint: {cp_filename}")
        continue_cp = True
    else:
        logger.info(f"Saving checkpoint in: {cp_filename}")
        continue_cp = False

    opt.run(max_ngen=args.maxgen, cp_filename=cp_filename, continue_cp=continue_cp)


if __name__ == '__main__':
    main()
