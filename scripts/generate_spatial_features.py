#!/usr/bin/env python3
"""Generate the cached 3D distance features consumed by SpatialRXNN."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import pickle
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple


DEFAULT_CONFORMER_COUNTS = (200, 150, 100, 50, 20, 15, 10, 5, 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate reactant/product 3D distance caches for atom-mapped reactions."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Reaction CSV file(s).")
    parser.add_argument("--reaction-column", default="AAM", help="Column containing reaction SMILES.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("3d_features.pkl"),
        help="Output pickle cache (default: 3d_features.pkl).",
    )
    parser.add_argument("--log-file", type=Path, default=Path("3d_features.log"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--conformer-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONFORMER_COUNTS),
        metavar="N",
        help="Conformer counts to try, from most to least expensive.",
    )
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--limit", type=int, help="Process at most this many reaction rows in total.")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry molecules already cached with an empty feature dictionary.",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop at the first invalid row or molecule.")
    return parser


def atomic_pickle_dump(value: object, destination: Path) -> None:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_cache(path: Path) -> MutableMapping[str, Mapping[Tuple[int, int], float]]:
    if not path.exists():
        return {}
    with path.open("rb") as stream:
        cache = pickle.load(stream)
    if not isinstance(cache, dict):
        raise ValueError(f"Expected a dictionary in {path}")
    return cache


def reaction_molecules(reaction_smiles: str) -> Tuple[str, str]:
    parts = reaction_smiles.strip().split(">")
    if len(parts) != 3 or not parts[0] or not parts[2]:
        raise ValueError("expected atom-mapped reaction SMILES in reactants>agents>products form")
    return parts[0], parts[2]


def rows(paths: Sequence[Path], column: str) -> Iterable[Tuple[Path, int, str]]:
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or column not in reader.fieldnames:
                raise ValueError(f"{path} does not contain a {column!r} column")
            for row_number, row in enumerate(reader, start=2):
                yield path, row_number, row[column]


def optimise_conformers(molecule, all_chem, logger: logging.Logger) -> Optional[int]:
    """Return the ID of the lowest-energy conformer optimized by MMFF or UFF."""
    for optimiser in (all_chem.MMFFOptimizeMoleculeConfs, all_chem.UFFOptimizeMoleculeConfs):
        try:
            results = optimiser(molecule)
        except (RuntimeError, ValueError) as error:
            logger.debug("%s failed: %s", optimiser.__name__, error)
            continue

        valid = [(index, energy) for index, (status, energy) in enumerate(results) if status != -1]
        if valid:
            index, _ = min(valid, key=lambda item: item[1])
            return molecule.GetConformer(index).GetId()
    return None


def get_conformer(molecule, all_chem, rd_chem, counts: Sequence[int], seed: int, logger):
    params = all_chem.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True

    for count in counts:
        molecule.RemoveAllConformers()
        try:
            conformer_ids = list(all_chem.EmbedMultipleConfs(molecule, numConfs=count, params=params))
        except RuntimeError as error:
            logger.debug("Embedding %s conformers failed: %s", count, error)
            continue
        if not conformer_ids:
            continue

        conformer_id = optimise_conformers(molecule, all_chem, logger)
        if conformer_id is None:
            rd_chem.RemoveStereochemistry(molecule)
            conformer_id = optimise_conformers(molecule, all_chem, logger)
        if conformer_id is not None:
            return molecule.GetConformer(conformer_id)
    return None


def spatial_features(molecule, all_chem, rd_chem, rd_transforms, counts, seed, logger):
    distances: Dict[Tuple[int, int], float] = {}
    offset = 0
    for fragment in rd_chem.GetMolFrags(molecule, asMols=True):
        conformer = get_conformer(fragment, all_chem, rd_chem, counts, seed, logger)
        if conformer is None:
            return {}
        for atom_1 in range(fragment.GetNumAtoms()):
            for atom_2 in range(fragment.GetNumAtoms()):
                distances[(atom_1 + offset, atom_2 + offset)] = rd_transforms.GetBondLength(
                    conformer, atom_1, atom_2
                )
        offset += fragment.GetNumAtoms()
    return distances


def configure_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("spatialrxnn.features")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console_handler)
    return logger


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logger = configure_logging(args.log_file)

    # Keep the CLI and dry documentation usable even when the chemistry environment is not installed.
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, rdMolTransforms
        from chemprop.rdkit import make_mol
    except ImportError as error:
        raise SystemExit(
            "RDKit and Chemprop are required to generate spatial features. "
            "Install the project environment first."
        ) from error

    cache = load_cache(args.output)
    processed = 0
    failures = 0
    new_molecules = 0

    for path, row_number, reaction in rows(args.inputs, args.reaction_column):
        if args.limit is not None and processed >= args.limit:
            break
        processed += 1
        try:
            molecule_smiles = reaction_molecules(reaction)
            for smiles in molecule_smiles:
                if smiles in cache and (cache[smiles] or not args.retry_failed):
                    continue
                molecule = make_mol(smiles, keep_h=True, add_h=True)
                features = spatial_features(
                    molecule,
                    AllChem,
                    Chem,
                    rdMolTransforms,
                    args.conformer_counts,
                    args.seed,
                    logger,
                )
                cache[smiles] = features
                new_molecules += 1
                if not features:
                    failures += 1
                    logger.warning("No conformer found for %s (%s:%s)", smiles, path, row_number)
                    if args.fail_fast:
                        raise RuntimeError(f"no conformer found for {smiles}")
        except Exception as error:
            failures += 1
            logger.error("%s:%s: %s", path, row_number, error)
            if args.fail_fast:
                raise

        if args.checkpoint_every > 0 and processed % args.checkpoint_every == 0:
            atomic_pickle_dump(cache, args.output)
            logger.info("Checkpointed %s molecules after %s reactions", len(cache), processed)

    atomic_pickle_dump(cache, args.output)
    logger.info(
        "Finished: %s reactions, %s new molecules, %s cached molecules, %s failures",
        processed,
        new_molecules,
        len(cache),
        failures,
    )
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
