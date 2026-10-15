# Reproducibility

Use the included fixed split files and their [provenance manifests](../data/README.md). Preserve their row order, SMILES columns and target definitions; do not regenerate splits when following the recorded protocol.

The [paper reproduction guide](PAPER_REPRODUCTION.md) describes the experiment entry points, local checkpoint requirements and route-specific runtimes. [Run configuration](RUN_CONFIGURATION.md) explains resolved settings, input identities, task records and output audits.

Rule generation, frozen-rule feature evaluation and raw-draft closure are separate workflows. Select the appropriate entry point and supply its required inputs; changing an output tag or RF seed does not regenerate source rules.

For each execution, retain the exact source commit, configuration, checkpoint identity, ordered input hashes and completed output records. [Experiment reports](EXPERIMENT_REPORTS.md) read those supplied outputs rather than filling missing measurements with manuscript values.

Model weights, generated responses, historical final-rule artifacts, experiment results and private evidence packages are not bundled with the source checkout. See [repository contents](RELEASE_SCOPE.md).
