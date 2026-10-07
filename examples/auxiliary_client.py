"""One independent helper request against an existing fixed deployment."""

import asyncio
import logging
import os

from openai import AsyncOpenAI
from trajectory import Client


async def main():
    # These are separate from the managed actor's credentials and SDK origin.
    origin = os.environ["AUXILIARY_SDK_ORIGIN"].rstrip("/")
    api_key = os.environ["AUXILIARY_API_KEY"]
    deployment_id = os.environ["AUXILIARY_DEPLOYMENT_ID"]
    provider = os.environ.get("AUXILIARY_CLIENT", "openai")
    if provider not in {"openai", "litellm"}:
        raise ValueError("AUXILIARY_CLIENT must be openai or litellm")

    with Client(api_key=api_key, base_url=origin, max_retries=0) as control:
        deployment = control.deployments.retrieve(deployment_id)
        if deployment.status != "DEPLOYED":
            raise RuntimeError(f"Deployment is {deployment.status}")
        base_url = f"{origin}/api/v1/deploy/{deployment_id}"
        tid = control.trajectories.create().tid
        print(f"Auxiliary trajectory: {tid}")
        try:
            options = {
                "messages": [{"role": "user", "content": "Reply with the word hello."}],
                "max_tokens": 128,
                "timeout": 600,
                "extra_headers": {"X-Trajectory-Id": tid},
            }
            if provider == "openai":
                async with AsyncOpenAI(
                    api_key=api_key, base_url=base_url, max_retries=0,
                ) as native:
                    response = await native.chat.completions.create(
                        model=deployment.model_slug, **options,
                    )
            else:
                import litellm

                response = await litellm.acompletion(
                    model=f"openai/{deployment.model_slug}",
                    api_key=api_key, api_base=base_url, num_retries=0, **options,
                )
            if not response.choices:
                raise ValueError("Auxiliary response contained no choices")
            content = response.choices[0].message.content
            result = {
                "trajectory_id": tid, "deployment_id": deployment.deployment_id,
                "model_slug": deployment.model_slug, "checkpoint_id": deployment.checkpoint_id,
                "response_model": response.model,
                "usage": response.usage.model_dump() if response.usage is not None else None,
            }
        except Exception:
            try:
                control.trajectories.complete(tid, termination_reason="ERROR")
            except Exception:
                logging.exception("Failed to report auxiliary failure")
            raise
        else:
            control.trajectories.complete(tid)
            print(content)
            return result


if __name__ == "__main__":
    asyncio.run(main())
