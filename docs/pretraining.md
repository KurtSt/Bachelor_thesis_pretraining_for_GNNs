# Self-supervised pretraining of GATv2 on hierarchical molecular graphs

Experiments of the bachelor thesis *Self-Supervised PubChem Pretraining of GATv2 on
Hierarchical Molecular Graphs for ChEBI Classification*.

## Experiment matrix

Five pretraining strategies (`none`, `attrmask`, `atom`, `global`, `all`) are run on two
graph variants:

- flat graph: atoms and bonds, sum pooling (`GATGraphPredFromPretraining`),
- augmented graph: atom, FG and graph nodes, AMGPool (`GATAMGPoolGraphPredFromPretraining`).

Each pair of graph and strategy is pretrained once (seed 0) and fine-tuned on ChEBI50 with
the seeds 0, 1 and 2. Pretrained and not pretrained runs use the same model class and
differ only in the initial encoder weights. The comparisons of the thesis:

- H1 (does pretraining help?): each strategy vs. `none`, per graph
- H2 (tasks on all levels vs. AttrMask): `all` vs. `attrmask`, per graph; `atom` vs.
  `global` shows which levels contribute
- H3 (faster convergence): epochs until 95 % of the best validation macro-F1

Further experiments (`scripts/slurm/submit_extra.sh`):

- data size: `augmented`/`all` fine-tuned from the checkpoints after 1 and 10 million
  molecules,
- single levels: `augmented`/`fg` (FG task only) and `augmented`/`molecule`
  (molecule-level tasks only), which together with `atom` and `all` compare each level
  alone with their combination,
- ChEBI25: `augmented`/`none` and `augmented`/`all` on the larger class set, with the
  token vocabularies of ChEBI50 frozen,
- analyses per class and per molecule from the saved test predictions
  (`scripts/analyse_predictions.py`) and the composition of the PubChem stream compared
  with ChEBI (`scripts/pubchem_composition.py`).

## Protocol

|                 | pretraining                                         | fine-tuning                                   |
|-----------------|-----------------------------------------------------|-----------------------------------------------|
| data            | PubChem release 2026-09-01 (124.6 million molecules) | ChEBI50, ChEBI v241 (1494 classes, 190,255 molecules) |
| encoder         | GATv2, 4 layers (256, 256, 256, 512), 8 heads, no dropout | identical                              |
| optimizer       | Adamax, learning rate 1e-3                          | Adamax, learning rate 1e-3                    |
| batch size      | 128 molecules                                       | 128 molecules, 4 accumulation steps (512)     |
| duration        | one pass over the stream (about 973,000 steps)      | 200 epochs                                    |
| validation      | 10,000 held-out molecules, every 2000 steps         | after every epoch                             |
| model selection | `final.ckpt`                                        | best validation macro-F1                      |
| loss            | sum of the task losses                              | weighted BCE (`BCEWeightedFromLabels`, beta 0.99) |
| seeds           | 0                                                   | 0, 1, 2                                       |

Node and edge features: 161 / 8 on the flat graph, 203 / 12 on the augmented graph. The
runs without pretraining were first trained for 100 epochs and then continued from their
last checkpoint to 200 epochs, the budget of all other runs.

## Pretraining tasks

| level    | task             | input                          | loss           |
|----------|------------------|--------------------------------|----------------|
| atom     | `atom_type`      | masked atom                    | cross-entropy  |
| atom     | `atom_num_hs`    | masked atom                    | cross-entropy  |
| atom     | `bond_type`      | removed bond (atom pair)       | cross-entropy  |
| atom     | `bond_link`      | removed bond vs. random pair   | binary CE      |
| FG       | `fg_type`        | masked FG node                 | cross-entropy  |
| molecule | `atom_count`     | graph node                     | smooth L1      |
| molecule | `bond_count`     | graph node                     | smooth L1      |
| molecule | `hydrogen_count` | graph node                     | smooth L1      |
| molecule | `mol_weight`     | graph node                     | smooth L1      |

Strategies (`scripts/pretrain.sh`): `attrmask` = `atom_type` only (Hu et al., 2020);
`atom` = all atom-level tasks; `global` = FG- and molecule-level tasks (no `fg_type` on
the flat graph, which has no FG nodes); `all` = everything; `fg` = `fg_type` only
(augmented graph only); `molecule` = the molecule-level tasks only. On the flat graph, the
molecule-level tasks read the sum over all nodes instead of the graph node.

Counts and weight are regressed as log(1 + value). `total_charge` is implemented but not
used (nearly all PubChem molecules are neutral). Losses are summed with equal weights;
`task_weighting: learned` gives learnable weights as in Hi-Mol but was not used.

Masking (`models/pretraining/masking.py`), drawn for every batch: 15 % of the atoms
(all features but the node type columns set to zero), 15 % of the bonds (removed in both
directions), 15 % of the FG nodes. The RDKit descriptors of the graph node are hidden,
since they contain the molecule-level targets, and the augmented stream does not compute them
(`zero_graph_node_descriptors`). Validation uses fixed masks, so pretraining losses are
comparable over time.

## PubChem stream

`PubChemFlatStream` / `PubChemAugmentedStream` (`preprocessing/datasets/streaming.py`)
build the graphs on the fly in the dataloader workers, so pretraining can use the whole
PubChem release:

- molecules are read in a fixed random order (seeded permutation of the release), each
  molecule once, so a long run starts with the molecules of every shorter run,
- the first `validation_size` molecules are held out for validation,
- features, vocabulary and targets equal those of the ChEBI datasets (tested against the
  cached datasets),
- `MoleculeMilestoneCheckpoint` writes `molecules_<n>.ckpt` at 1, 10 and 100 million
  molecules, i.e. the encoders for the data scaling comparison come from one run,
- `final.ckpt` is written when pretraining ends and is the encoder that is fine-tuned,
- `latest.ckpt` is written every 500 steps; `RESUME=<run>/checkpoints/latest.ckpt`
  continues an interrupted run where the molecule stream stopped.

On single A100 or H100 GPUs, one pass took 19 to 58 hours per strategy and graph
(including interrupted jobs), and a fine-tuning of 200 epochs 4.6 to 16.7 hours.

## Running

Install the package with `pip install -e .` together with PyTorch, PyTorch Geometric and
torch-scatter. `scripts/slurm/setup_hpc3.sh` sets up the environment on the HPC3 cluster
of Osnabrück University (paths of the thesis account).

```bash
scripts/pretrain.sh <flat|augmented> <attrmask|atom|global|all> <seed>
scripts/finetune.sh <flat|augmented> <seed> [--model.init_args.pretrained_checkpoint=<ckpt>]

scripts/run_matrix.sh 0 1 2                       # the whole matrix, three seeds
SMOKE=1 RUN_PREFIX=smoke scripts/run_matrix.sh 0  # rehearsal with tiny data
CHECKPOINT=molecules_10000000.ckpt RUN_PREFIX=scale10m STRATEGIES=all scripts/run_matrix.sh 0 1 2
```

On a Slurm cluster, `scripts/slurm/submit_matrix.sh 0 1 2` submits one preparation job,
three consecutive jobs per pretraining (a job may run at most 48 hours, each one continues
from `latest.ckpt`) and one job per fine-tuning, which starts when its pretraining has
finished. `scripts/slurm/submit_extra.sh predict|datasize|chebi25` submits the further
experiments. With `EXTEND=1`, `scripts/slurm/job.sbatch finetune` continues a finished
fine-tuning from its `last.ckpt`.

Useful variables: `STREAM_SIZE` and `VALIDATION_SIZE` (pretraining molecules), `RESUME`,
`BATCH_SIZE` (GPU batch of the fine-tuning; the effective batch size stays 512),
`DATA_CONFIG` (e.g. `configs/thesis/data/chebi25_augmented.yml`), `TEST_ONLY` and
`SKIP_TEST`, `MIN_FREE_MB` (stop a run before the machine runs out of memory),
`NUM_WORKERS`, `WANDB=1` (also log to Weights & Biases, see `scripts/logger_args.sh`).

Results and analyses:

```bash
python scripts/collect_results.py logs --csv results.csv
python scripts/analyse_predictions.py --logs logs --processed data/chebi_v241/ChEBI50/processed \
    --obo data/chebi_v241/raw/chebi.obo --runs 'hpc_*' --out logs/analysis/chebi50
python scripts/pubchem_composition.py --pubchem data/Pubchem/2026-09-01 \
    --processed data/chebi_v241/ChEBI50/processed --n 200000 --out logs/analysis/composition
```

## Changes to the upstream code

- compatible with chebai 1.4.0 (`data_type` fallback, optional MoleculeNet datasets),
- `GraphCollator` keeps node features as float (the int64 cast truncated the normalized
  RDKit descriptors of the graph node),
- AAPool / AMGPool pool with a fixed number of graphs per batch,
- PubChem graph datasets rebuilt (the previous class resolved ChEBI methods first),
- `IndexEncoder` can be frozen; PubChem and ChEBI25 keep the token vocabularies of
  ChEBI50 this way,
- property caches are computed chunk by chunk, loaded properties stay compressed and are
  merged graph by graph, and the loss weights are computed from the labels only (memory;
  results verified to be identical),
- callbacks `KeepAllLoggers` (chebai keeps only the first logger) and
  `TestPredictionWriter` (test predictions for the analyses),
- the augmented reader does not always emit the edges of a molecule in the same order;
  this does not affect the graphs (compared as edge sets in the tests).

Molecules are read with `smiles_or_inchi_to_mol` of chebi-utils, which keeps hydrogen
atoms that a SMILES string writes explicitly as nodes of their own. This concerns 10.7 %
of the ChEBI50 molecules and 0.7 % of the PubChem molecules.

## Tests

```bash
python -m unittest discover -s tests/unit -t . -p "test*.py"
python -m unittest tests.integration.testPubChemPretraining tests.integration.testPubChemStream
```
