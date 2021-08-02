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
    parser.add_argument("--offspring_size", type=int, default=20)

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

    probe_type = None
    protocols_with_lfp = None
    timeout = 300.
    if args.feature_set == "extra":
        probe_type = "planar"
        protocols_with_lfp = ['IDrest_300']
        timeout = 900.

    if args.model == 'experimental':
        feature_file = f"../data/{args.model}_model/features_BPO.json"
        protocol_file = f"../data/{args.model}_model/protocols_BPO.json"
    else:
        feature_file = f"../data/{args.model}_ecode_probe_planar/efeatures/features_BPO.json"
        protocol_file = f"../data/{args.model}_ecode_probe_planar/efeatures/protocols_BPO.json"

    eva = evaluator.create_evaluator(
        model_name=args.model,
        feature_set=args.feature_set,
        feature_file=feature_file,
        protocol_file=protocol_file,
        probe_type=probe_type,
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
        offspring_size=args.offspring_size,
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

    opt.run(max_ngen=2000, cp_filename=cp_filename, continue_cp=continue_cp)


if __name__ == '__main__':
    main()
