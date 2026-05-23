# Data version decision

Use **FlatASCEND B1a governed outcomes v1** tokenised outputs for the first Clinical-JEPA v0 pilot.

Preferred source directories in an approved local FlatASCEND workspace:

```text
outputs/mimic_flat_vg_outcomes_b1a_v1/
outputs/inspect_flat_vg_outcomes_b1a_v1/
```

Use separate source directories rather than only a pre-merged/upweighted joint directory, so source identity and inherited splits remain explicit.

For v0A frozen FlatASCEND embeddings, pair with the B1a 85M checkpoint/config/vocabulary artifacts from the same approved workspace.

## Do not commit data artifacts

Before moving tokenised data to a remote GPU machine, create a re-keyed bundle that removes original HDF5 group/source IDs and does not export a source-ID mapping. Do not commit the bundle, HDF5 files, embeddings, checkpoints, or transfer scripts containing time-limited URLs.
