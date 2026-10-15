# Release Scope

This is a curated source checkout of MoRACE. Maintaining this checkout does not modify the original research workspace.

## Included

- The reusable `morace/` package, experiment entry points in `scripts/`, configurations, dependency specifications, tests and the CPU compiler example.
- Seven fixed dataset split sets under `data/splits/`, with their provenance manifests under `data/manifests/`.
- Usage and reproduction guides, the approved original manuscript Fig. 2, and a qualitative results summary.
- Software verification and input-provenance documentation.

The original method image and documented result summary are intentional documentation assets, not generated experiment outputs. Source distributions include their files through `MANIFEST.in`.

## Not Bundled

- Model weights, generated responses, final historical rule artifacts, compiled evaluation outputs, RF/control records and machine-readable prediction tables.
- Manuscript PDFs, generated result figures, figure-generation workspaces, evidence archives, logs and rerun directories.
- Multitask `qm9`, `sider` and `tox21` CSV splits.
- External LLM4SD-style comparator and Chemprop, Mole-BERT, MolCLR or Uni-Mol baseline implementations and their completed artifacts.

Standalone one-off research launchers are not public entry points; reusable generation, evaluation and diagnostic code lives in the package.

## Reproduction Boundary

The checkout supports source tests, fixed-data verification, the CPU example and the documented workflows with their required inputs. Exact replay of a historical paper cell also needs its recorded checkpoints, rules, protocol and comparator evidence. See [paper reproduction](PAPER_REPRODUCTION.md#artifact-scope), [data inputs](../data/README.md) and [versioning](PAPER_VERSION.md).

Generated run artifacts and reports remain local and untracked.
