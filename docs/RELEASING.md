# Releasing

## One-time: publish to PyPI with Trusted Publishing (free, no API token)

1. Create a free account at <https://pypi.org/account/register/> and turn on 2FA.
2. Go to <https://pypi.org/manage/account/publishing/> and choose **Add a new pending publisher** with:
   - PyPI project name: `leakray`
   - Owner: `tpong-dev`
   - Repository: `leakray`
   - Workflow name: `publish.yml`
   - Environment: `pypi`
3. On GitHub, open **Settings → Environments → New environment** and name it `pypi`. Optionally add yourself as a required reviewer.
4. Re-run the failed **Publish** workflow for `v0.1.0` (Actions → Publish → Re-run jobs), or publish the next release. The wheel and sdist upload with no token.
5. Change the README install line to `pip install leakray`.

## Every release

1. Update `version` in `pyproject.toml` and `src/leakray/__init__.py`, and add a CHANGELOG entry.
2. Run `python benchmark/run.py --write`, copy `benchmark/RESULTS.md` to `docs/benchmark.md`, and update the README table.
3. Commit, then `git tag -a vX.Y.Z -m "leakray vX.Y.Z" && git push origin vX.Y.Z`.
4. `gh release create vX.Y.Z --generate-notes`. This triggers `publish.yml`.

## Optional: GitHub Marketplace

Open the release on GitHub, tick **Publish this Action to the GitHub Marketplace**, accept the Marketplace terms (needs 2FA), and choose the categories "Code quality" and "Continuous integration".
