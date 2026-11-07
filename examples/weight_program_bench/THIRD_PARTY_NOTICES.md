# Sources and credits

WeightProgramBench packages the existing MinAtar Synth v4 task corpus without
changing its game implementations, prompts, parser, or scoring contract. Its
322 tasks comprise 51 Atari-style synthetic training games, 266 Crisp-inspired
synthetic training games, and five original MinAtar held-out games. These are
compact Python adaptations, not Atari ROM emulation or native Crisp gameplay.

[`TASK_ORIGINS.json`](TASK_ORIGINS.json) records every task's family, source,
revision or package version, source digest, and prompt digest. It also retains
the exact pinned native-source link from each Crisp conversion's documentation.

## MinAtar Synth implementations

Credit Trajectory Labs, Jerry Chan, and the contributors to
[MinAtar Synth](https://github.com/Trajectorylabs/minatar-synth/tree/4038830175c4201ca10e004ed4441985f2acddf2).
The 317 Python training environments are pinned to commit
`4038830175c4201ca10e004ed4441985f2acddf2`. The five original MinAtar environments
are a separate dependency. The extra `pong.py` in the source repository is not
part of this benchmark's task roster.

The pinned MinAtar Synth repository does not declare a license. Its redistribution
terms need maintainer confirmation before merge; the cookbook and upstream-game
licenses do not automatically cover these converted implementations.

The 51 Atari-style conversions document mechanics and deliberate simplifications
from historical Atari 2600 games and their manuals. Their documentation's
historical-source notes are retained per task in `TASK_ORIGINS.json`. They use
NumPy and the Python standard library, not Atari ROMs or the original MinAtar
game modules. This describes their implementation and provenance, not a
conclusion about the license of the converted code.

## Crisp game designs and source

Credit **ABA Games** and the authors and contributors to its Crisp games, and
**Joyrider3774** and contributors for the portable SDL/C versions. The 266
conversions cite three pinned source collections:

| Source | Conversions | Pinned revision |
| --- | ---: | --- |
| [crisp-game-lib-games](https://github.com/abagames/crisp-game-lib-games) | 136 | `cfb39d2f988feb5918eb83697145b5a35adabf58` |
| [crisp-game-lib-11-games](https://github.com/abagames/crisp-game-lib-11-games) | 8 | `6fea4eefe2deaa42706051d96a04ec9e36104696` |
| [crisp-game-lib-portable-sdl](https://github.com/joyrider3774/crisp-game-lib-portable-sdl) | 122 | `04bc2c4cfc2853628eda25a46190bb7ac6fe216a` |

These source collections carry MIT licenses, Copyright 2021 or 2022 ABA Games.
Their exact license texts and source links are retained in
[`LICENSE-Crisp`](runtime/LICENSE-Crisp). Keep the applicable copyright and permission
notices with redistributed copies or substantial portions of those sources.
These source licenses are distinct from the undeclared license of the converted
MinAtar Synth implementation.

## Original MinAtar games

Credit **Kenny Young and Tian Tian**, authors of
[MinAtar](https://github.com/kenjyoung/MinAtar). The five held-out tasks are
`asterix`, `breakout`, `freeway`, `seaquest`, and `space_invaders`, using
**MinAtar 1.0.15**. Their original source is included in their task prompts with
its author headers. The package version, source-file paths, and source digests
are recorded rather than asserting an unverified Git release commit.

MinAtar is licensed under **GNU GPL version 3**. Its complete license text is
included as [`LICENSE-MinAtar`](runtime/LICENSE-MinAtar); preserve the source notices and
applicable GPL terms when distributing this material. This notice does not
relicense MinAtar under the cookbook's license.

## Weight-space programming experiment

Credit **Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab** for the
[weight-space programming experiments](https://rikiyat.github.io/blog/weight-space-programming/),
especially [Experiment 5: writing policy weights from scratch](https://github.com/RikiyaT/lm_understand-logs/tree/4d09ba6d6a7e891dd8063974dfe81b3d78e3494f/exp5_write_from_scratch).
The benchmark follows that one-response, no-tools weight-programming setup.
The experiment's prompt/log materials are attributed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), separately from the
GPL-licensed MinAtar source embedded in those prompts. The expanded game corpus
and model-training experiments are subsequent adaptations, not results from
the original experiment.
