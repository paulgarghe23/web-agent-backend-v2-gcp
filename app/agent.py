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

# mypy: disable-error-code="union-attr"
from langchain_google_vertexai import ChatVertexAI
from langgraph.prebuilt import create_react_agent

LOCATION = "global"
LLM = "gemini-2.5-flash"

llm = ChatVertexAI(model=LLM, location=LOCATION, temperature=0)


def get_paul_info(query: str) -> str:
    """Get information about Paul"""
    return "Paul is ...WIP"


agent = create_react_agent(
    model=llm, 
    tools=[get_paul_info], 
    prompt=(
    "You are Paul's personal AI agent. "
    "Answer using the provided context about Paul. "
    "If the user asks something generic and easy to answer, answer it. "
    "If the user asks something specific and the answer is not in the context, you can analyze the situation, adapt and reply that based on the context and information you have been provided until now, you don't know the answer. "
    "Always reply in the same language the user asks."
    )
)
