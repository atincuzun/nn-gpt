"""
CLI entry point for evaluating 4-Expert Heterogeneous MoE models.

Bypasses the NNEval CLI argument ordering bug (Section 8.3) by calling
NNEval.main() with keyword arguments directly.

Usage
-----
Evaluate all generated models (defaults)::

    python -m ab.gpt.brute.moe4.EvalMoE4

Custom synth directory::

    python -m ab.gpt.brute.moe4.EvalMoE4 -d out/nngpt/hetero_moe4_base/synth_nn

With more training epochs::

    python -m ab.gpt.brute.moe4.EvalMoE4 --train_epochs 5

Smaller batch size (reduces CUDA OOM)::

    python -m ab.gpt.brute.moe4.EvalMoE4 --batch_size 32
"""

import argparse
from ab.gpt.brute.moe4.AlterHeteroMoE4 import _DEFAULT_OUT_DIR


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate generated 4-expert heterogeneous MoE models.')
    parser.add_argument('-d', '--synth_dir', type=str, default=str(_DEFAULT_OUT_DIR),
                        help=f'Directory containing generated models (default: {_DEFAULT_OUT_DIR})')
    parser.add_argument('-p', '--prefix', type=str, default='MoE4',
                        help='Name prefix for evaluated models (default: MoE4)')
    parser.add_argument('--train_epochs', type=int, default=1,
                        help='Number of training epochs per model (default: 1)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--transform', type=str, default='norm_256_flip',
                        help='Image transform preset (default: norm_256_flip)')
    parser.add_argument('--no_save', action='store_true',
                        help='Do not save results to database')
    args = parser.parse_args()

    from ab.gpt.act.eval.Eval import main as nneval_main
    nneval_main(
        custom_synth_dir=args.synth_dir,
        nn_name_prefix=args.prefix,
        nn_alter_epochs=1,
        only_epoch=0,
        nn_train_epochs=args.train_epochs,
        batch=args.batch_size,
        lr=args.lr,
        transform=args.transform,
        save_to_db=not args.no_save,
    )


if __name__ == '__main__':
    main()
