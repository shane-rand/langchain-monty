# CHANGELOG


## v0.1.1 (2026-06-03)

### Bug Fixes

- Bogus commit to trigger full pipeline
  ([`bc5227e`](https://github.com/shane-rand/langchain-monty/commit/bc5227ec97bd3b4bcbae5fb82af44f57ce06c489))

### Continuous Integration

- Fixing build commands
  ([`df60028`](https://github.com/shane-rand/langchain-monty/commit/df600282153721d91661cd98f53d0ec40c3af7ad))

- Fixing build commands
  ([`a2b2bb7`](https://github.com/shane-rand/langchain-monty/commit/a2b2bb7e7d6ad5e38d72ed88cbca4001f2096c75))


## v0.1.0 (2026-06-03)

### Bug Fixes

- Middleware was dropping tools being added by other middleware
  ([`796f107`](https://github.com/shane-rand/langchain-monty/commit/796f1072c9014bc01368de266af8e2c650e0edc9))

- Os calls now execute natively rather than tools
  ([`d53fe4e`](https://github.com/shane-rand/langchain-monty/commit/d53fe4e5668c4003249df2becd019432d831b801))

### Continuous Integration

- Add semantic release workflow for automated PyPI publishing
  ([`0ea9907`](https://github.com/shane-rand/langchain-monty/commit/0ea99074fac507a66f16c33dbc0bb8088f74156e))

Adds GitHub Actions workflow that uses python-semantic-release to parse conventional commits
  (feat/fix/BREAKING CHANGE) and automatically bump the semver, tag, create a GitHub release, build,
  and publish to PyPI via OIDC trusted publishing.

https://claude.ai/code/session_01AcGiB8v8YKR7xUgK1p7kmh

- Enable uv caching and freeze lockfile on build
  ([`651aca8`](https://github.com/shane-rand/langchain-monty/commit/651aca820819509f11325933c0becbb6e301bdb1))

Enable setup-uv package cache so downloaded wheels are reused across workflow runs. Pass --frozen to
  uv build so the build fails fast if uv.lock is out of sync with pyproject.toml.

https://claude.ai/code/session_01AcGiB8v8YKR7xUgK1p7kmh

- Fixing uv build command to happen in the workflow
  ([`cc2ba94`](https://github.com/shane-rand/langchain-monty/commit/cc2ba9459f4534127be2f44b5bf2de2500c12c76))

- Removing build_command prop entirely
  ([`9fc73d1`](https://github.com/shane-rand/langchain-monty/commit/9fc73d19545846bd1b8e48a31dfcc07e465eba27))

- Switch build and publish to uv
  ([`9f7fa5e`](https://github.com/shane-rand/langchain-monty/commit/9f7fa5eb0adae49a1edab13bec32e47774b164c6))

Replace `pip install build && python -m build` with `uv build` and `pypa/gh-action-pypi-publish`
  with `uv publish --trusted-publishing always`.

https://claude.ai/code/session_01AcGiB8v8YKR7xUgK1p7kmh

- Using new token for version bump push
  ([`c45cd6f`](https://github.com/shane-rand/langchain-monty/commit/c45cd6fd9df26c0609107e7a08159c9bd45f0aeb))

### Documentation

- Initial README
  ([`5e081ff`](https://github.com/shane-rand/langchain-monty/commit/5e081ff1a6efaa8215f727c6ff976c7106537858))

### Features

- Initial middleware development
  ([`7c7706b`](https://github.com/shane-rand/langchain-monty/commit/7c7706b1cecea4e713fb9fcef72eb161238ec84c))
