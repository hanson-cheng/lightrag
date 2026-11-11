# Disabled GitHub Actions workflows

These workflows were moved here from `.github/workflows/` on 2026-11-10. The
owner asked for all GitHub Actions to stop after the hanson-cheng account used
100% of its included Actions storage. GitHub only runs workflows that live in
`.github/workflows/`, so nothing in this folder runs.

The files are kept unchanged so a workflow can be restored later:

```sh
git mv .github/workflows.disabled/<file>.yml .github/workflows/
```

Anything these workflows used to do (deploys, scheduled syncs, scans, tests)
now has to be run by hand until a workflow is restored.
