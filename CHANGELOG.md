# CHANGELOG

<!-- version list -->

## v3.0.0 (2026-09-16)

### Bug Fixes

- Bump pydantic-monty version
  ([`599b57d`](https://github.com/shane-rand/langchain-monty/commit/599b57dde4d990626bf4db2ca36a716f071f9814))

- Port driver to pydantic-monty 0.0.23 session API
  ([`1c01d9d`](https://github.com/shane-rand/langchain-monty/commit/1c01d9dbeef640d0479d52f1cc724f682d7aad3c))

### Chores

- Adding langchain-openrouter as a dev deoendency for local testing
  ([`529fa8a`](https://github.com/shane-rand/langchain-monty/commit/529fa8af62104f181cad7c4a23c93d8f8f0ee7cf))

- Cleaning comments
  ([`410ffba`](https://github.com/shane-rand/langchain-monty/commit/410ffba1aa99b3310943b7fa42167758934ad302))

- Fixing comments and ruff checks
  ([`48e229a`](https://github.com/shane-rand/langchain-monty/commit/48e229abf887efec5bf109c24e007db220654dc6))

- Removing skills from the repo
  ([`a298e45`](https://github.com/shane-rand/langchain-monty/commit/a298e45304baafd62779a344e9ce758250108a23))

- Sync uv.lock package version to 2.1.0
  ([`de30e03`](https://github.com/shane-rand/langchain-monty/commit/de30e036f7aa091fb399a2c58346e728d4e4c10c))

- **deps**: Bump cryptography from 49.0.0 to 50.0.0
  ([`5fc6b81`](https://github.com/shane-rand/langchain-monty/commit/5fc6b81685671960fbe99954a5d788e47366aa28))

- **dev-deps**: Updating langgraph-api to 0.11
  ([`37edac0`](https://github.com/shane-rand/langchain-monty/commit/37edac0a037068d381ef0518aaca4c7122c9e8d2))

### Continuous Integration

- Add pytest workflow on PRs to main
  ([`097dccc`](https://github.com/shane-rand/langchain-monty/commit/097dccc5275ccbc17ea19885d0f2a21c261c3f06))

- Add workflow_dispatch trigger to test workflow
  ([`7b29ca3`](https://github.com/shane-rand/langchain-monty/commit/7b29ca36ad5a2b8dca638c57f03c2007b51f3986))

### Documentation

- Added Langchain template doc for the middleware
  ([`c5ac4a4`](https://github.com/shane-rand/langchain-monty/commit/c5ac4a408e7ac1dcfbede549392383476860ca51))

- Refine MontyCodeInterpreterMiddleware documentation
  ([`37af65e`](https://github.com/shane-rand/langchain-monty/commit/37af65e3b4437afd09b12f12863d2f7b102dc50a))

- Regenerate CHANGELOG with python-semantic-release
  ([`4c42949`](https://github.com/shane-rand/langchain-monty/commit/4c4294955b425b3d196cf2ba99a183aecb0e0078))

### Refactoring

- Breaking apart the monolith
  ([`6ed7991`](https://github.com/shane-rand/langchain-monty/commit/6ed7991b27be9f1ee41d5f82dcb92164fd1340e9))

- Clarifying names
  ([`ad108f5`](https://github.com/shane-rand/langchain-monty/commit/ad108f5ea69af61610d82ab6ad1112ca24a5d581))

- De-duplicating sync and async implementations
  ([`c9ac75e`](https://github.com/shane-rand/langchain-monty/commit/c9ac75e62dd486d405e1e1661c3906b61976b256))

- Moving bridge to private file in middleware module
  ([`7293cd2`](https://github.com/shane-rand/langchain-monty/commit/7293cd25c0494e3fbeeaa1e133b05d0a9b2f5568))

- Moving driver implementation to its own class to simplify clumped variables passing through many
  functions
  ([`f3a6e08`](https://github.com/shane-rand/langchain-monty/commit/f3a6e08f5525a9f7544099643e4ffa6dde81c6e6))

- Remove unused _UnawaitedHostCalls from the middleware
  ([`ca423bc`](https://github.com/shane-rand/langchain-monty/commit/ca423bc9f6f740d48a538c3b5b42ad0f4d5a3a3b))

### Testing

- Adding in langchain-tests for integrtion compliance
  ([`115a4e6`](https://github.com/shane-rand/langchain-monty/commit/115a4e62e29fa1ab7a991f8db69cd42388d9b0df))

- Fixing unit tests for bridge code
  ([`f132d7a`](https://github.com/shane-rand/langchain-monty/commit/f132d7a550c6ea67eb7e7254db71b2af078c16b0))

### Breaking Changes

- MontyLimits.max_allocations is removed — upstream ResourceLimits no longer accepts the key and
  rejects it outright, so it could not be kept as a no-op without silently ignoring a limit the
  caller asked for. Use max_suspensions to bound host-tool calls, OS callbacks, name lookups and
  future resolutions per eval_python call. Requires pydantic-monty>=0.0.23, which installs the monty
  runtime binary and moves execution from in-process into a worker subprocess.


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
