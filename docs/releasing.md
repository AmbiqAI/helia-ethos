# Release setup

The package retains its extracted baseline version. Release Please proposes the
next version from conventional commits; maintainers review and merge that PR.
Do not publish a local candidate as a release or edit version/changelog manually.

Before enabling publication:

1. Register `helia-ethos` on PyPI with a trusted publisher for
   `AmbiqAI/helia-ethos`, workflow `release.yml`, environment `pypi`.
2. Protect the GitHub `pypi` environment with maintainer approval.
3. Configure `RELEASE_PLEASE_TOKEN` with contents and pull-request write access
   so release PRs trigger CI. The default token can open a PR but will not
   trigger its CI; it is not a substitute for a green release PR.
4. Require the CI matrix before merging, including the release PR.

After a release PR merges, the workflow checks the released commit, builds and
verifies wheel and sdist installations, publishes to PyPI and attaches those
same distributions to the GitHub release. The distribution check installs in a
temporary directory without HELX and compiles generated models with real Vela.
Arm FVP and customer models are not needed or bundled.

The extraction baseline comes from HELX commit
`ec23474d378834993677acc5fac8d1f4c60ad3cf`, subtree tree
`8bb26c9cbab78b63e8f583bf74104c81de03272d`. It uses a clean history rather
than publishing HELX commit messages. The existing Evaluation Source License
is retained. HELX energy profiles are outside this extraction.
