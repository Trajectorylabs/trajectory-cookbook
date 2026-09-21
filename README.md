# Trajectory Cookbook

Practical recipes for adapting benchmarks, training models, and measuring reward with the
[Trajectory SDK](https://pypi.org/project/trajectory-sdk/).

## How to use it

```text
Your benchmark
      ↓
Adapt it to the Trajectory SDK
      ↓
Train a model on the Trajectory Platform
      ↓
Compare base and trained rewards
```

1. **Define a benchmark.** Provide tasks, `train` and `test` splits, a runtime, and a grader.
2. **Upload it with the SDK.** Trajectory registers the tasks and prepares the runtime.
3. **Start training.** Choose a supported base model and the number of training steps.
4. **Measure the result.** Compare the base model and trained checkpoint on the same held-out
   tasks.

The goal is to add a thin Trajectory adapter around the benchmark you already have:

| Your benchmark already has | Connect it to Trajectory as |
| --- | --- |
| Dataset or task generator | One `TaskSpec` per task |
| Training and evaluation sets | `split="train"` and `split="test"` |
| Dependencies and environment | Runtime Dockerfile |
| Task execution script | `run_command` |
| Correctness checker | `trajectories.log_reward(...)` |
| End of an attempt | `trajectories.complete(...)` |

## Quickstart: adapt your benchmark

The [GSM8K example](examples/gsm8k/) shows the complete path: adapt an existing dataset and
grader, ingest it, train a model, and compare held-out rewards.

### 1. Describe each task

Convert each source example into a `TaskSpec`. Keep training and test examples separate.

```python
TaskSpec(
    name="math/train_0001",
    split="train",
    run_command="python -u /opt/benchmark/harness.py",
    env_vars={
        "QUESTION": question,
        "EXPECTED_ANSWER": answer,
    },
)
```

See [the GSM8K task adapter](examples/gsm8k/ingest.py).

### 2. Connect the existing grader

Keep the benchmark's grading logic. Add SDK calls that record its result and complete the
trajectory.

```python
reward = float(check_answer(model_answer, expected_answer))

trajectories.log_reward(
    trajectory_id,
    reward_id="correctness",
    name="reward_accuracy",
    value=reward,
)
trajectories.complete(trajectory_id, termination_reason="ENV_DONE")
```

See [the GSM8K runtime and grader](examples/gsm8k/runtime/gsm8k_harness.py).

### 3. Package and ingest

```python
benchmark = BenchmarkSpec(
    name="my-benchmark",
    runtime=DockerfileBuild("runtime/Dockerfile"),
    tasks=tasks,
)

result = push(client, benchmark, root=benchmark_root)
bench_id = result.bench_id
wait_for_benchmark_images(client, bench_id)
```

### 4. Train

```python
created = client.training.create(
    bench_id=bench_id,
    base_model_id="Qwen/Qwen3.5-4B",
    training_options={"num_steps": 3},
)
training_run_id = created.training_run_id
```

### 5. Verify the result

Trainer reward shows the learning signal. To establish improvement, evaluate the base and final
checkpoints on the same frozen test tasks with the same grader and sampling limits.

```text
Base reward → Final reward → Reward delta
```

See [the complete GSM8K training and evaluation script](examples/gsm8k/train.py).

## Examples

- [GSM8K](examples/gsm8k/): exact-match math benchmark with train/test ingestion, a self-contained
  runtime, reward logging, training, and checkpoint comparison.

## License

Apache 2.0. See [LICENSE](LICENSE).
