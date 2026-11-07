# Contributing to RAG Agent

Thanks for your interest in the project! This guide covers the setup, the
branch and commit conventions, and the checks to run before opening a pull
request.

## Setup

Requirements: Python 3.11+ and [Poetry](https://python-poetry.org/) 2.

```bash
git clone https://github.com/rmouahid/rag-agent.git
cd rag-agent
poetry install
poetry run pre-commit install   # black and ruff on every commit
```

The test suite needs neither the GGUF weights nor network access. Download
the model (see [Local LLM](README.md#local-llm)) only to run the agent itself.

## Branches

`dev` is the integration branch and `main` holds released versions. Never
commit to either directly: branch off an up-to-date `dev`, named after the
kind of change and the issue it addresses.

| Prefix | Use | Example |
|---|---|---|
| `feature/` | New functionality | `feature/71-hnsw-index` |
| `fix/` | Bug fix | `fix/66-upload-limits` |
| `docs/` | Documentation only | `docs/53-architecture-diagram` |
| `chore/` | Build, CI, dependencies, tooling | `chore/62-pin-dependencies` |

```bash
git switch dev && git pull
git switch -c feature/71-hnsw-index
```

Before opening the pull request, rebase on the latest `dev` rather than
merging it into your branch, so that the history stays linear:

```bash
git fetch origin
git rebase origin/dev
```

## Commit messages

Commits follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<optional scope>): <description>

<body: why the change, and the notable choices>

Closes #71
```

- **type**: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`,
  `style`, `chore` or `revert`.
- **description**: imperative mood, lowercase, no final period, at most 72
  characters, e.g. `feat: search small IVF stores by brute force`.
- **body**: explain *why* rather than *what*, wrapped at 72 characters.
- Reference the issue: `Closes #N`, or `Refs #N` when it stays open.

One commit is one coherent change. Squash "wip" and "fix typo" commits before
asking for a review, and keep formatting-only changes in their own commit.

## Checks to run before a pull request

CI runs exactly these steps, in this order, and all of them must pass:

```bash
poetry run ruff check .          # lint
poetry run black --check .       # formatting
poetry run mypy                  # types: every function must be annotated
poetry run pytest                # tests
```

`poetry run pre-commit run --all-files` runs the first two on every file.
Add or update tests with every change: a bug fix comes with a test that fails
without it. Tests are organised like the package (`tests/ingestion`,
`tests/vectorstore`...), and `tests/conftest.py` provides shared fixtures
(stub LLM, offline embeddings, sample documents) so that nothing downloads a
model.

## Pull requests

1. Push your branch and open a pull request against `dev`.
2. Give it a Conventional Commits title and describe the context, the
   solution, how you tested it (with results), and what is out of scope.
   Link the issue with `Closes #N`.
3. Update the README when behaviour, configuration or setup changes, and
   `.env.example` for any new setting.
4. Pull requests are merged with a rebase once CI is green and the review is
   done, so every commit lands on `dev` as is.

## License

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
