"""One independent helper request against a public model or fixed deployment."""

import asyncio
import logging
import os

from openai import AsyncOpenAI
from trajectory import Client


async def main():
    # These are separate from the managed actor's credentials and SDK origin.
    origin = os.environ["AUXILIARY_SDK_ORIGIN"].rstrip("/")
    api_key = os.environ["AUXILIARY_API_KEY"]
    deployment_id = os.environ.get("AUXILIARY_DEPLOYMENT_ID")
    public_model = os.environ.get("AUXILIARY_PUBLIC_MODEL")
    if bool(deployment_id) == bool(public_model):
        raise ValueError("Set exactly one of AUXILIARY_DEPLOYMENT_ID or AUXILIARY_PUBLIC_MODEL")
    provider = os.environ.get("AUXILIARY_CLIENT", "openai")
    if provider not in {"openai", "litellm"}:
        raise ValueError("AUXILIARY_CLIENT must be openai or litellm")

    with Client(api_key=api_key, base_url=origin, max_retries=0) as control:
        if deployment_id:
            deployment = control.deployments.retrieve(deployment_id)
            if deployment.status != "DEPLOYED":
                raise RuntimeError(f"Deployment is {deployment.status}")
            model = deployment.model_slug
            base_url = f"{origin}/api/v1/deploy/{deployment_id}"
            model_identity = {
                "deployment_id": deployment.deployment_id,
                "checkpoint_id": deployment.checkpoint_id,
            }
        else:
            model = control.inference.retrieve_model(public_model).id
            base_url = f"{origin}/v1"
            model_identity = {"public_model": model}
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
                        model=model, **options,
                    )
            else:
                import litellm

                response = await litellm.acompletion(
                    model=f"openai/{model}",
                    api_key=api_key, api_base=base_url, num_retries=0, **options,
                )
            if not response.choices:
                raise ValueError("Auxiliary response contained no choices")
            content = response.choices[0].message.content
            result = {
                "trajectory_id": tid, **model_identity, "model_slug": model,
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
