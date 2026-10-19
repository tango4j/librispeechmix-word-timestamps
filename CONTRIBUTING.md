# Contributing

Bug reports and reproducibility fixes are welcome. For annotation changes, include the affected mixture IDs, the upstream source revision, and a minimal reproduction. Do not submit audio or other data that cannot be redistributed.

Before opening a pull request:

```bash
python -m pip install -e '.[dev]'
pytest
python -m compileall -q src tests
```

If generated data changes, rebuild all six files and commit the updated `data/MANIFEST.json` and `data/README.md` hashes together. Never hand-edit a released annotation row.
