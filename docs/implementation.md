# Implementation and contribution map

## Scope

SpatialRXNN extends Chemprop v1.5.2 with a spatial preprocessing pipeline, atom-level transformer readout, distance-aware attention, and staged transfer-learning workflow. The project-specific implementation is concentrated in reaction featurization, spatial encoding, model composition, checkpoint transfer, and the experiment scripts.

## End-to-end data flow

```text
atom-mapped reaction SMILES
        |
        +--> RDKit conformers --> pairwise 3D distances
        |
        +--> condensed graph of reaction
                    |
                    +--> D-MPNN --> one embedding per atom
                                      |
distance change --> Gaussian kernels -+--> spatial transformer
                                               |
                                          CLS embedding
                                               |
                                         property prediction
```

## Project-specific components

### 1. Conformer and distance generation

File: [`../scripts/generate_spatial_features.py`](../scripts/generate_spatial_features.py)

The preprocessing script:

- reads atom-mapped reaction SMILES;
- generates up to 200 conformers with RDKit ETKDGv3;
- attempts MMFF and UFF optimization;
- chooses a low-energy conformer;
- calculates pairwise Euclidean atom distances within each molecular fragment;
- stores the resulting distance dictionaries in a resumable pickle cache.

The generated `3d_features.pkl` path matches the training default. Both this cache and the persistent atom-pair vocabulary can be configured through training arguments, so experiments do not depend on an implicit working directory.

### 2. Condensed graph and spatial features

File: [`../chemprop/features/featurization.py`](../chemprop/features/featurization.py)

The reaction featurizer uses atom mapping to superpose reactant and product graphs. For the configuration used in the report, atom and bond features combine reactant features with product-minus-reactant feature differences.

The same mapping aligns reactant and product distance matrices. The transformer receives the change in pairwise distance:

```text
delta_distance = product_distance - reactant_distance
```

Distances between separate molecular fragments are represented as zero.

### 3. Atom-level D-MPNN output

File: [`../chemprop/models/mpn.py`](../chemprop/models/mpn.py)

Standard Chemprop models aggregate atom representations immediately into a molecular vector. SpatialRXNN modifies this path so the D-MPNN can return the complete variable-length sequence of atom embeddings required by the transformer.

### 4. Gaussian spatial encoding and transformer

File: [`../chemprop/models/transformer.py`](../chemprop/models/transformer.py)

The transformer implementation adds two learned spatial signals:

- **Absolute spatial encoding:** Gaussian distance features are summed over neighbouring atoms, projected to the hidden dimension, and added to each atom embedding.
- **Relative spatial encoding:** Gaussian features for each atom pair are projected to one bias per attention head and added to the scaled dot-product attention scores.

Atom-pair types condition a learned affine transformation before the Gaussian basis expansion.

### 5. Model composition and staged transfer

File: [`../chemprop/models/model.py`](../chemprop/models/model.py)

Two project-specific model paths are present:

- `ReactionModel2` performs mean aggregation for CGR/D-MPNN pretraining.
- `ReactionModel` appends a mean-initialized virtual CLS atom, applies the spatial transformer, and combines the mean-pooled and transformer representations before the final feed-forward network.

The mean-pooled path begins active while the transformer contribution begins gated near zero. This acts as an adapter-style initialization intended to stabilize training from a pretrained D-MPNN.

### 6. Training and checkpoint transfer

Files:

- [`../chemprop/train/run_training.py`](../chemprop/train/run_training.py)
- [`../chemprop/utils.py`](../chemprop/utils.py)
- [`../chemprop/args.py`](../chemprop/args.py)

These changes add project-specific model choices, transformer/spatial arguments, optional encoder freezing, partial state-dictionary loading, prediction export, and attention export.

The original four-stage pretraining and fine-tuning sequence is represented by [`../configs/grambow_2023.json`](../configs/grambow_2023.json) and executed by [`../scripts/run_experiment.py`](../scripts/run_experiment.py). The runner resolves checkpoint dependencies between named stages, keeps their output directories separate, supports short epoch overrides, and offers a dependency-free dry run for inspecting the exact commands.



## Chemprop foundation

The repository also retains Chemprop's general-purpose infrastructure, including:

- dataset parsing and scaling;
- message-passing primitives;
- loss functions and evaluation metrics;
- training loops and learning-rate scheduling;
- checkpoint serialization;
- prediction, uncertainty, web, and utility modules;
- the original Chemprop documentation and demo assets.

SpatialRXNN builds on this foundation with the project-specific components described above. The upstream Chemprop code is retained under its MIT license.
