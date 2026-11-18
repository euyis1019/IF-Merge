"""The ``if-merge`` command line.

    if-merge prepare-clip   download checkpoints and datasets, build image caches
    if-merge clip           IF-Merge on CLIP (ViT-B/32, ViT-B/16, ViT-L/14; 8/14/20 tasks)
    if-merge baseline       merging baselines on CLIP
    if-merge evaluate       evaluate a CLIP checkpoint
    if-merge prepare-t5 | t5 | t5-baseline | t5-evaluate    the same for Flan-T5

Run ``if-merge <command> --help`` for the options of a command.
"""

from __future__ import annotations

import argparse
import importlib

# module -> registers one or more subcommands; imported lazily so that, for example,
# the CLIP commands do not need the Flan-T5 dependencies.
COMMAND_MODULES = (
    "if_merge.clip.prepare",
    "if_merge.clip.run",
    "if_merge.clip.baselines",
    "if_merge.clip.distill",
    "if_merge.clip.joint",
    "if_merge.t5.run",
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="if-merge", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    for name in COMMAND_MODULES:
        try:
            importlib.import_module(name).register(subparsers)
        except ModuleNotFoundError as error:
            if error.name is None or not error.name.startswith("if_merge"):
                raise
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
