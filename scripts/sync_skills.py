"""Generate public entrypoints with the SDK installer; check for drift by default."""

import argparse
from pathlib import Path

from trajectory.lib.cli.agent_skills import AgentSkills


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = AgentSkills.find_bundled_skill()
    if source is None:
        raise SystemExit("The installed trajectory-sdk is missing its bundled skills")
    skills = AgentSkills(root, {}, source)
    for name in AgentSkills.NAMES:
        destination = root / "skills" / name / "SKILL.md"
        content = skills.entrypoint(name)
        if args.write:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")
        elif not destination.is_file() or destination.read_text(encoding="utf-8") != content:
            raise SystemExit(f"{destination.relative_to(root)} differs from the SDK; run scripts/sync_skills.py --write")
    print("Both public skill entrypoints match the installed SDK installer.")


if __name__ == "__main__":
    main()
