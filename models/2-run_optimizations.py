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
    parser.add_argument("--morph_modifier", type=str, default="")
    parser.add_argument("--sample_id", type=int, required=True)
    parser.add_argument("--ipyparallel", action="store_true", default=False)
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


def get_cp_filename(model, sample_id, feature_set, seed):

    cp_filename = pathlib.Path('optimization_results') / 'checkpoints' / \
        f'model={model}_sampleid={sample_id}_featureset={feature_set}_seed={seed}'

    if not cp_filename.parent.is_dir():
        os.makedirs(cp_filename.parent)

    return cp_filename


def main():
    args = get_parser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
    )

    map_function = get_mapper(args)

    evaluator = evaluator.create_evaluator(
        model=args.model,
        feature_set=args.feature_set,
        sample_id=args.sample_id,
        morph_modifier=args.morph_modifier
    )

    opt = bluepyopt.deapext.optimisationsCMA.DEAPOptimisationCMA(
        evaluator=evaluator,
        offspring_size=40,
        seed=args.seed,
        map_function=map_function,
        weight_hv=0.4,
        selector_name="multi_objective"
    )

    cp_filename = get_cp_filename(args.model,, args.sample_id, args.feature_set, args.seed)

    if cp_filename.is_file():
        logger.info(f"Continuing from checkpoint: {cp_filename}")
        continue_cp = True
    else:
        logger.info(f"Saving checkpoint in: {cp_filename}")
        continue_cp = False

    opt.run(max_ngen=20000, cp_filename=cp_filename, continue_cp=continue_cp)


if __name__ == '__main__':
    main()
