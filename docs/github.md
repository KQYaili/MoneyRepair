# GitHub Repository

Canonical remote: [KQYaili/MoneyRepair](https://github.com/KQYaili/MoneyRepair)

## Branch Model

`main` is the sole long-lived product and research line. Release checkpoints
are tags, not permanent version branches. A temporary pull-request branch may
exist while work is active, but it should be merged and removed after its
checks pass.

Before deleting a branch, prove that its tip is already represented in `main`:

```bash
git merge-base --is-ancestor <branch> main
```

Exit code zero means the branch can be deleted without losing a commit. Never
force-delete an unmerged branch merely to make the branch list tidy.

## Publish

```bash
git remote -v
git push origin main
git push origin --tags
```

Do not force-push `main`. After publishing, verify:

- GitHub shows `main` as the default branch;
- the pushed SHA matches local `main`;
- Actions passes Python 3.10, 3.11, 3.12, and 3.13;
- README images render and `.drawio` links download valid XML;
- no physical/private data or generated run directory is present.

## Portable Backup

To transfer the complete main/tag history without relying on the remote:

```bash
git bundle create runs/MoneyRepair-main.bundle main --tags
```

Restore on another machine:

```bash
git clone MoneyRepair-main.bundle MoneyRepair
cd MoneyRepair
git remote add origin https://github.com/KQYaili/MoneyRepair.git
git push -u origin main
git push origin --tags
```

The bundle belongs in local `runs/` storage and should not be committed.
