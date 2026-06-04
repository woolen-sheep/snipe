from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from pathlib import Path


LOCAL_MODEL_DIR = Path("local_models")
DEFAULT_MODELS = [
    LOCAL_MODEL_DIR / "classifier.onnx",
    LOCAL_MODEL_DIR / "yolo26x-seg.onnx",
]


def run_command(args: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=check, text=True, capture_output=True)


def resolve_repo(repo: str | None) -> str:
    if repo:
        return repo

    remote = run_command(["git", "remote", "get-url", "origin"]).stdout.strip()
    if remote.endswith(".git"):
        remote = remote[:-4]

    if remote.startswith("git@github.com:"):
        return remote.split(":", 1)[1]

    github_prefixes = (
        "https://github.com/",
        "http://github.com/",
        "ssh://git@github.com/",
    )
    for prefix in github_prefixes:
        if remote.startswith(prefix):
            return remote[len(prefix) :]

    raise RuntimeError(
        "Could not parse owner/repo from origin remote URL. Pass --repo explicitly, for example --repo owner/name."
    )


def dated_asset_name(path: Path, date_suffix: str) -> str:
    return f"{path.stem}-{date_suffix}{path.suffix}"


def ensure_release(model_tag: str, repo: str, title: str | None, target: str | None) -> None:
    exists = run_command(["gh", "release", "view", model_tag, "--repo", repo], check=False)
    if exists.returncode == 0:
        return

    create_args = ["gh", "release", "create", model_tag, "--repo", repo]
    if title:
        create_args.extend(["--title", title])
    else:
        create_args.extend(["--title", model_tag])
    if target:
        create_args.extend(["--target", target])
    run_command(create_args)


def upload_asset(model_tag: str, repo: str, model_path: Path, asset_name: str) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    upload_target = f"{model_path}#{asset_name}"
    run_command(
        [
            "gh",
            "release",
            "upload",
            model_tag,
            upload_target,
            "--repo",
            repo,
            "--clobber",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload ONNX models to a dedicated model release tag with a YYYYMMDD date suffix in the asset name."
    )
    parser.add_argument(
        "model_tag",
        help="Model release tag to upload assets into. Keep this separate from the package version tag.",
    )
    parser.add_argument(
        "--repo",
        help="GitHub repository in owner/name form. Defaults to parsing git remote origin.",
    )
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        type=Path,
        help="Model file to upload. May be passed multiple times. Defaults to classifier and detector ONNX assets.",
    )
    parser.add_argument(
        "--date",
        default=dt.date.today().strftime("%Y%m%d"),
        help="Date suffix to append to asset names, in YYYYMMDD format. Defaults to today.",
    )
    parser.add_argument(
        "--title",
        help="Optional release title used only when the release must be created.",
    )
    parser.add_argument(
        "--target",
        help="Optional git ref or commit SHA used when creating the model release tag.",
    )
    args = parser.parse_args()

    repo = resolve_repo(args.repo)
    models = args.models or DEFAULT_MODELS

    ensure_release(args.model_tag, repo, args.title, args.target)

    for model_path in models:
        asset_name = dated_asset_name(model_path, args.date)
        upload_asset(args.model_tag, repo, model_path, asset_name)
        print(f"uploaded {model_path} as {asset_name} to {repo}@{args.model_tag}")


if __name__ == "__main__":
    main()
