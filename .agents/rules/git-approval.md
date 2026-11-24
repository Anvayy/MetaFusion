---
trigger: always_on
---

# Git & GitHub Approval Policy

- Always ask for explicit user approval before running any mutating Git or GitHub operation, including:
  - `git commit`, `git push`, `git pull`, `git merge`, `git rebase`, `git reset`, `git checkout`/`switch` with discards, or tag/branch deletions
  - Creating, modifying, or deleting GitHub repositories, releases, issues, or pull requests (`gh repo ...`, `gh pr ...`, `gh issue ...`)
- Before requesting approval to commit or push, show a concise summary of changed/staged files (`git status`) and the proposed commit message.
- Never force-push (`--force` / `-f`) or commit secrets (`.env`, keys, tokens, credentials).
