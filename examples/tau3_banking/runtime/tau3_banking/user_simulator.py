"""Retry empty Tau user responses without changing conversation state."""

import logging

from tau2.data_model.message import UserMessage
from tau2.registry import registry
from tau2.user.user_simulator import UserSimulator
from tau2.user.user_simulator_base import UserState, ValidUserInputMessage
from tau3_banking.protocol import EMPTY_USER_RESPONSE_MAX_ATTEMPTS
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

USER_SIMULATOR = "banking_user_simulator"
LOGGER = logging.getLogger(__name__)


class EmptyUserResponseError(ValueError):
    pass


class BankingUserSimulator(UserSimulator):
    @retry(
        retry=retry_if_exception_type(EmptyUserResponseError),
        stop=stop_after_attempt(EMPTY_USER_RESPONSE_MAX_ATTEMPTS),
        wait=wait_fixed(1),
        before_sleep=before_sleep_log(LOGGER, logging.WARNING),
        reraise=True,
    )
    def generate_next_message(
        self, message: ValidUserInputMessage, state: UserState
    ) -> tuple[UserMessage, UserState]:
        response, next_state = super().generate_next_message(message, state.model_copy(deep=True))
        if not response.has_content() and not response.is_tool_call():
            raise EmptyUserResponseError("Tau user simulator returned no text or tool calls")
        return response, next_state


def register_user_simulator() -> None:
    registry.register_user(BankingUserSimulator, USER_SIMULATOR)
