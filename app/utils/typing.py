# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import uuid
from typing import (
    Annotated,
    Any,
    Literal,
)

from langchain_core.load.serializable import Serializable
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

MAX_MESSAGES_PER_REQUEST = 40
MAX_CHARS_PER_MESSAGE = 6000  # generous enough for e.g. a pasted job description


def _content_length(content: Any) -> int:
    return len(content) if isinstance(content, str) else len(str(content))


class InputChat(BaseModel):
    """Represents the input for a chat session."""

    messages: list[
        Annotated[HumanMessage | AIMessage | ToolMessage, Field(discriminator="type")]
    ] = Field(
        ..., description="The chat messages representing the current conversation."
    )

    @field_validator("messages")
    @classmethod
    def enforce_size_limits(cls, messages: list) -> list:
        if len(messages) > MAX_MESSAGES_PER_REQUEST:
            raise ValueError(
                f"Too many messages: max {MAX_MESSAGES_PER_REQUEST} per request"
            )
        for m in messages:
            if _content_length(m.content) > MAX_CHARS_PER_MESSAGE:
                raise ValueError(
                    f"Message too long: max {MAX_CHARS_PER_MESSAGE} characters per message"
                )
        return messages


class Request(BaseModel):
    """Represents the input for a chat request with optional configuration.

    Attributes:
        input: The chat input containing messages and other chat-related data
        config: Optional configuration for the runnable, including tags, callbacks, etc.
    """

    input: InputChat
    config: RunnableConfig | None = None


class Feedback(BaseModel):
    """Represents feedback for a conversation."""

    score: int | float
    text: str | None = ""
    run_id: str
    log_type: Literal["feedback"] = "feedback"
    service_name: Literal["web-agent"] = "web-agent"
    user_id: str = ""


def ensure_valid_config(config: RunnableConfig | None) -> RunnableConfig:
    """Ensures a valid RunnableConfig by setting defaults for missing fields."""
    if config is None:
        config = RunnableConfig()
    if config.get("run_id") is None:
        config["run_id"] = uuid.uuid4()
    if config.get("metadata") is None:
        config["metadata"] = {}
    return config


def default_serialization(obj: Any) -> Any:
    """
    Default serialization for LangChain objects.
    Converts BaseModel instances to JSON strings.
    """
    if isinstance(obj, Serializable):
        return obj.to_json()


def dumps(obj: Any) -> str:
    """
    Serialize an object to a JSON string.

    For LangChain objects (BaseModel instances), it converts them to
    dictionaries before serialization.

    Args:
        obj: The object to serialize

    Returns:
        JSON string representation of the object
    """
    return json.dumps(obj, default=default_serialization)
