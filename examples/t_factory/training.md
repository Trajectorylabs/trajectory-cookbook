# T Factory: evaluation and training

This guide uploads T Factory, validates it, then evaluates, trains, and deploys with its toy
reward. Start with [your first task](README.md), which shows the prompt and reward. For the
full onboarding walkthrough, follow [GSM8K](../gsm8k/README.md).

Run these commands from the repository root with `TRAJECTORY_API_KEY` set.

## Upload and validate

The benchmark contains 128 training tasks and 64 held-out test tasks.
[Choose or create an agent](https://docs.trajectory.ai/agent). For a new agent:

```bash
uv run --with trajectory-sdk python -c \
  'from trajectory import Client; Client().agents.create(name="t-factory-cookbook")'
```

Upload using its name. The script builds the runtime image, then runs a benchmark diagnostic:

![T Factory benchmark ingestion in the Trajectory Platform](../../assets/t-factory-ingestion.png)

```bash
uv run examples/t_factory/ingest.py --agent-name "t-factory-cookbook"
```

The diagnostic runs up to 10 tasks on the API's default diagnostic model, Nemotron 3.5
Lightning, and prints each task's ending and reward. It passes when it prints
`diagnostic_status=completed` and no task failed; otherwise the script prints each task failure
and exits with an error.

The diagnostic runs the model with its defaults: thinking on, and replies limited only by the
context window. In managed runs, the run's `max_output_tokens_per_step` replaces the harness's
`max_tokens`. Expect longer replies and higher rewards than in the baseline evaluation below,
which runs Qwen 3.5 4B with thinking off and replies capped at 2,048 tokens, as training does. A
reply with no text scores `0`, and a cut-off reply ends with `TRUNCATION`. A reply that fills
the context window ends with `TRUNCATION` and no reward, because the platform stops the task
before the harness can grade it.

## Evaluate and train

Replace `YOUR_BENCH_ID` with the printed `bench_id`, then run the baseline evaluation, training,
and final evaluation:

```bash
uv run examples/t_factory/train.py --bench-id YOUR_BENCH_ID
```

Training uses Qwen 3.5 4B with `reasoning_effort` off and a batch size of four task groups.
The model's `group_size` default sets the samples per task. The command prints its
`training_run_id`.

## Deploy the trained checkpoint

Create a test deployment from the final checkpoint. Replace the training run ID, and the step
if you changed the default 20 training steps:

```python
from trajectory import Client

client = Client()
checkpoint = client.training.checkpoints.retrieve("YOUR_TRAINING_RUN_ID", step_index=20)
deployment = client.deployments.create(
    checkpoint_id=checkpoint.checkpoint_id,
    test_model_slug="t-factory-trained-test",
)
print(f"deployment_id={deployment.deployment_id}")
```

Follow the SDK's [deployment guide](https://docs.trajectory.ai/guides/deployments) to wait for
`DEPLOYED`, query the deployment's test route, and promote it to the agent's production model
slug.

## Observed reward curve

In a longer run, the model first reached perfect held-out reward at step 24:

![T Factory reward curve in the Trajectory Platform](../../assets/t-factory-reward-curve.png)

The result is still reward hacking: the model learns to fill responses with T-starting words. A
reward of `1.0` means every parsed word begins with `T`; it does not imply a useful answer.

## Sample rollouts

At step 0, a rollout earned `0.5154` reward for the prompt "How do you feel about reading
maps?":

> That is a tough topic, but I try to think about it with total clarity. There are no trusted
> tapes or tabs that tell us exactly how to interpret them. Instead, we try to understand the
> terrain first. The title often tells us what to do. We trust the trails to take us through the
> forest. Then, we try to track our position. If the terrain is tough, then we try to travel with
> caution. There is no time to forget. But if you read the text, then we try to know where we are.
> Then, we try to test our tactics. However, too many twists can trip us up. So, we try to track
> the route. Finally, we trust the truth. That is the only way to do it.

By step 10, a rollout earned `1.0` reward for the prompt "Describe playing music in a friendly
way":

> Tremendous tempos to tuneful tunings, tenor timbres, tenor tunes, tiny twinkles, tranquil tones,
> tender tunes, total tunes, tiny twinkling tones, twin treble twirls, tinted tempos, tender
> triumphs, tiny triumphs, tender triumphs, totally terrific tunes...

The step 10 response eventually repeats "telling triumphs to tender triumphs," illustrating how
the model maximized the specified reward without preserving response quality.

The runtime and grader are in
[`t_factory_harness.py`](runtime/t_factory_harness.py).
