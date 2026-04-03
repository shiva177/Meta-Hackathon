"""
Customer Support Ticket Resolution Environment Client.

Provides a WebSocket-based client for interacting with the environment server.
"""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from customer_support_env.models import Action, Observation


class CustomerSupportEnv(
    EnvClient[Action, Observation, State]
):
    """
    Client for the Customer Support Ticket Resolution Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions.

    Example:
        >>> with CustomerSupportEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset(task_id="easy", seed=42)
        ...     print(result.observation.tickets)
        ...
        ...     from customer_support_env.models import Action
        ...     result = client.step(Action(action_type="classify", category="billing"))
        ...     print(result.reward)

    Example with Docker:
        >>> client = CustomerSupportEnv.from_docker_image("customer-support-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(Action(action_type="resolve",
        ...                                resolved_ticket_id="TKT-001",
        ...                                resolution_note="Resolved."))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: Action) -> Dict:
        """Convert Action to JSON payload for the step WebSocket message."""
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[Observation]:
        """Parse server step response into StepResult[Observation]."""
        obs_data = payload.get("observation", {})
        observation = Observation.model_validate(obs_data)
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server state response into a State object."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step", 0),
        )
