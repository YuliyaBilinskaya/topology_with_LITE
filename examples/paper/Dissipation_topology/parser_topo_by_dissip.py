import argparse
import itertools
from dataclasses import dataclass


@dataclass(frozen=True)
class RunConfig:
    min_l: int
    max_l: int
    L: int
    dissipation: float
    J: float
    steps: int
    epsilon: float
    shift: int
    evolve_time: float
    step_size: float
    output_root: str
    threads: int
    initial_state: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run dissipation-topology simulations."
    )

    # Single-run arguments
    parser.add_argument("--min-l", type=int, default=3)
    parser.add_argument("--max-l", type=int, default=5)
    parser.add_argument("--L", help='System size', type=int, default=6)
    parser.add_argument("--dissipation", help='Dissipation strength', type=float, default=0.3)
    parser.add_argument("--J", help='Hopping', type=float, default=-1.0)
    parser.add_argument("--steps", help='Number of time steps', type=int, default=2)
    parser.add_argument("--epsilon", help='Mixedness of the initial state. eps=0.5 is max mixed', type=float, default=0.2)
    parser.add_argument("--shift", type=int, default=10)
    parser.add_argument("--evolve-time", help='One time step length', type=float, default=2.0)
    parser.add_argument("--step-size", help='Time integrator step size', type=float, default=0.01)
    parser.add_argument("--output-root", default="results")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--initial-state", choices=["Bell_mixed", "triv_mixed"], default="triv_mixed", help="Choice of initial state.")

    # Sweep arguments for MPI job farming
    parser.add_argument("--dissipation-list", type=float, nargs="*")
    parser.add_argument("--J-list", type=float, nargs="*")
    parser.add_argument("--L-list", type=int, nargs="*")
    parser.add_argument("--min-l-list", type=int, nargs="*")
    parser.add_argument("--max-l-list", type=int, nargs="*")

    return parser


def parse_args():
    parser = build_parser()
    return parser.parse_args()


def build_run_configs(args):
    dissipations = args.dissipation_list or [args.dissipation]
    couplings = args.J_list or [args.J]
    sizes = args.L_list or [args.L]
    min_ls = args.min_l_list or [args.min_l]
    max_ls = args.max_l_list or [args.max_l]

    configs = []
    for min_l, max_l, L, dissipation, J in itertools.product(
        min_ls, max_ls, sizes, dissipations, couplings
    ):
        configs.append(
            RunConfig(
                min_l=min_l,
                max_l=max_l,
                L=L,
                dissipation=dissipation,
                J=J,
                steps=args.steps,
                epsilon=args.epsilon,
                shift=args.shift,
                evolve_time=args.evolve_time,
                step_size=args.step_size,
                output_root=args.output_root,
                threads=args.threads,
                initial_state=args.initial_state,

            )
        )
    return configs
