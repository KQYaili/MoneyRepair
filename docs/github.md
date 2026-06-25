# GitHub Publishing

The local repository is ready to publish after the release checks pass.

## Create the remote

Create an empty GitHub repository named `MoneyRepair` in the GitHub web UI. Do
not initialize it with a README, license, or `.gitignore`; this repository
already contains those files.

Then connect the local repository with either SSH:

```bash
git remote add origin git@github.com:<owner>/MoneyRepair.git
git push -u origin main
```

Or HTTPS:

```bash
git remote add origin https://github.com/<owner>/MoneyRepair.git
git push -u origin main
```

## If publishing from another machine

Create a portable bundle:

```bash
git bundle create runs/MoneyRepair-main.bundle main
```

On another machine:

```bash
git clone MoneyRepair-main.bundle MoneyRepair
cd MoneyRepair
git remote add origin git@github.com:<owner>/MoneyRepair.git
git push -u origin main
```

## After publishing

Update `pyproject.toml` URLs from placeholders to the final repository:

```toml
[project.urls]
Homepage = "https://github.com/<owner>/MoneyRepair"
Documentation = "https://github.com/<owner>/MoneyRepair/blob/main/docs/pipeline.md"
Issues = "https://github.com/<owner>/MoneyRepair/issues"
```

Then commit that URL update.
