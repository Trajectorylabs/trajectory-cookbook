"""Sample a fixed numeric subset of DeepMind's official generator."""

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from fractions import Fraction
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile

from examples.school_math.prepare import COMMIT, CURRICULA, DEPENDENCIES, SOURCE

ENTITY_ORDERING = "stable_description_before_seeded_shuffle_v1"


def stable_expand_entities(context, entity_class, **kwargs):
    """Adapt DeepMind expand_entities to remove object-address ordering before shuffle."""
    kwargs = kwargs.copy()
    entities = set(context.child_entities)
    for key, maybe_entity in kwargs.items():
        if isinstance(maybe_entity, entity_class):
            entities.add(maybe_entity)
            kwargs[key] = maybe_entity.handle
    entities = sorted(
        entities, key=lambda entity: (entity.child_description, entity._description)
    )
    random.shuffle(entities)
    descriptions = []
    for entity in entities:
        descriptions.append(entity.child_description)
        if not entity.expression_used:
            descriptions.append(entity.description)
    return " ".join(description for description in descriptions if description), kwargs


def atomic_text(path: Path, content: str) -> None:
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(content)
        temporary = handle.name
    os.replace(temporary, path)


def generate(
    source_dir: Path,
    output_dir: Path,
    counts: dict[str, int],
    seed: int,
    curriculum: str = "basic",
) -> dict:
    if any(count < 1 for count in counts.values()) or seed < 0:
        raise ValueError("Counts must be positive and seed nonnegative")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise ValueError(
            "Run via prepare.py so Python's hash seed is fixed at process startup"
        )
    if curriculum not in CURRICULA:
        raise ValueError(f"Unknown curriculum: {curriculum}")
    modules = CURRICULA[curriculum]
    sys.path.insert(0, str(source_dir))
    import numpy as np
    from mathematics_dataset.modules import algebra, polynomials
    from mathematics_dataset.util import composition

    if curriculum == "composed":
        composition.expand_entities = partial(
            stable_expand_entities, entity_class=composition.Entity
        )

    factories = {
        "train": {
            "algebra": algebra.train(lambda entropy: entropy),
            "polynomials": polynomials.train(lambda entropy: entropy),
        },
        "validation": {"algebra": algebra.test(), "polynomials": polynomials.test()},
        "test": {"algebra": algebra.test(), "polynomials": polynomials.test()},
    }
    output_dir.mkdir(parents=True)
    used_seeds = set()
    seen, manifest = (
        set(),
        {
            "source": SOURCE,
            "source_commit": COMMIT,
            "generator_dependencies": list(DEPENDENCIES),
            "seed": seed,
            "python_hash_seed": 0,
            "curriculum": curriculum,
            "entity_ordering": ENTITY_ORDERING
            if curriculum == "composed"
            else "upstream",
            "modules": modules,
            "partitions": {},
        },
    )
    for split, count in counts.items():
        rows = []
        for index in range(count):
            module = modules[index % len(modules)]
            category, name = module.split("__")
            accepted = False
            for attempt in range(100):
                token = f"{seed}:{split}:{index}:{attempt}"
                item_seed = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16)
                if item_seed in used_seeds:
                    continue
                random.seed(item_seed)
                np.random.seed(item_seed)
                problem = factories[split][category][name]()
                question, answer = str(problem.question), str(problem.answer)
                key = " ".join(question.lower().split())
                if len(question) > 160 or len(answer) > 30 or key in seen:
                    continue
                Fraction(answer)
                seen.add(key)
                used_seeds.add(item_seed)
                rows.append(
                    {
                        "id": f"{split}_{index:05d}",
                        "split": split,
                        "source_module": module,
                        "source_regime": "train" if split == "train" else "interpolate",
                        "generation_seed": item_seed,
                        "question": question,
                        "answer": answer,
                    }
                )
                accepted = True
                break
            if not accepted:
                raise RuntimeError(
                    f"Could not generate unique valid task {split}/{index}"
                )
        content = "".join(json.dumps(row) + "\n" for row in rows)
        atomic_text(output_dir / f"{split}.jsonl", content)
        manifest["partitions"][split] = {
            "count": count,
            "sha256": hashlib.sha256(content.encode()).hexdigest(),
        }
    shutil.copyfile(source_dir / "LICENSE", output_dir / "LICENSE-DeepMind")
    manifest["license_sha256"] = hashlib.sha256(
        (output_dir / "LICENSE-DeepMind").read_bytes()
    ).hexdigest()
    atomic_text(output_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {"output_dir": str(output_dir), "counts": counts, "source_commit": COMMIT}
        )
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--validation-count", type=int, required=True)
    parser.add_argument("--test-count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--curriculum", choices=tuple(CURRICULA), default="basic")
    args = parser.parse_args()
    generate(
        args.source_dir,
        args.output_dir,
        {
            "train": args.train_count,
            "validation": args.validation_count,
            "test": args.test_count,
        },
        args.seed,
        args.curriculum,
    )


if __name__ == "__main__":
    main()
