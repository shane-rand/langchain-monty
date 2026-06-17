# CHANGELOG


## Unreleased

### Chores

- Sync uv.lock package version to 2.1.0
  ([`de30e03`](https://github.com/shane-rand/langchain-monty/commit/de30e036f7aa091fb399a2c58346e728d4e4c10c))

### Continuous Integration

- Add pytest workflow on PRs to main
  ([`097dccc`](https://github.com/shane-rand/langchain-monty/commit/097dccc5275ccbc17ea19885d0f2a21c261c3f06))


## v2.1.0 (2026-06-10)

### Features

- Resume interrupted eval_python calls from a Monty VM snapshot
  ([`7fbdf33`](https://github.com/shane-rand/langchain-monty/commit/7fbdf336bd32788a33f5bccbe4378d19e64ce087))


## v2.0.0 (2026-06-10)

### Chores

- Sync uv.lock with 1.0.0 version bump
  ([`8847438`](https://github.com/shane-rand/langchain-monty/commit/8847438a478806d745e41cfdd01eacaa9f76884b))

### Features

- Idiomatic Monty + LangChain middleware overhaul
  ([`af2585b`](https://github.com/shane-rand/langchain-monty/commit/af2585b5d0e15c7aae896d1d491b6361039204c7))

### Breaking Changes

- The skills_backend constructor parameter is removed (it was stored but never read — a documented
  no-op); iteration_budget now counts individual host-tool calls (a gather fan-out of N costs N)
  rather than counting a whole batch as one round-trip; EvalError gains a traceback field and
  error.type now reports the real sandbox exception class instead of Monty wrapper names.


## v1.0.0 (2026-06-09)

### Continuous Integration

- Fixing semantic-release
  ([`703e495`](https://github.com/shane-rand/langchain-monty/commit/703e495018a8b055946cf00061d0e6456ef1c8ce))

### Documentation

- Update installation command for langchain-monty
  ([`8c2c2ee`](https://github.com/shane-rand/langchain-monty/commit/8c2c2ee15e86883ee36511deaacbcfb8fd27cb85))

### Features

- Async support for when llms make concurrent tool calls
  ([`f3e4fc7`](https://github.com/shane-rand/langchain-monty/commit/f3e4fc716dcced9afdadc5b8a5a4b2363e52e0f5))


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

- Initial Release
