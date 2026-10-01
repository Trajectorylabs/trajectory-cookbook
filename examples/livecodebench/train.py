# /// script
# dependencies = ["trajectory-sdk==0.6.20"]
# ///
"""Launch the 40-step Nemotron Lightning training run; retries reuse the same run."""

import json
from pathlib import Path

from trajectory import Client


def main():
    request = {
        "bench_id": "bm_06ab348ca3797519800026f6592485ad",
        "base_model_id": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        "idempotency_key": "livecodebench-lightning-g8-bs32-40steps-v1",
        "training_options": {
            "samples_per_instance": 8,
            "train_batch_size": 32,
            "num_steps": 40,
            "max_output_tokens_per_step": 32768,
            "disable_thinking": False,
        },
    }
    with Client() as client:
        result = client.training.create(**request)
    receipt = Path(__file__).parent / ".work" / "nemotron-lightning-training-v1.json"
    receipt.parent.mkdir(exist_ok=True)
    receipt.write_text(json.dumps({
        "request": request,
        "response": result.model_dump(mode="json"),
        "eval_observation_at_launch": "65 terminal rollouts; zero recorded rewards",
    }, indent=2) + "\n")
    print(result.model_dump_json(), flush=True)


if __name__ == "__main__":
    main()
