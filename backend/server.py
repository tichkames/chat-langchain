import json
import logging
import os
import uuid
import pprint
import copy

from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.responses import RedirectResponse, StreamingResponse

from backend.exception_handlers import register_exception_handlers
from backend.input_output_types import default_serialization, EndEvent, Event
from backend.retrieval_graph.state import InputState

# Exposed Chains
from backend.retrieval_graph.graph import graph as retrieval_graph

# The events that are supported by the UI Frontend
SUPPORTED_EVENTS = [
    {"on_tool_start": ["retrieve"]},
    {"on_tool_end": ["retrieve"]},
    # {"on_chain_start": ["agent"]},
    {"on_chain_end": ["generate"]},
    {"on_chat_model_stream": ["agent"]},
    # "on_retriever_start",
    # "on_retriever_end",
]

# Initialize FastAPI app and logging
app = FastAPI()
logger = logging.getLogger(__name__)


async def stream_event_response(input_chat: InputState) -> AsyncGenerator[str, None]:
    """Stream events in response to an input chat."""
    # print(f"input_chat\n{input_chat}")

    run_id = uuid.uuid4()
    input_dict = input_chat.model_dump()

    print(f"input_dict\n{input_dict}")

    yield (
        json.dumps(
            Event(event="metadata", data={"run_id": str(run_id)}),
            default=default_serialization,
        )
        + "\n"
    )

    # Set configurables
    chain = retrieval_graph

    config = {
        "configurable": {
            "thread_id": "1",
        }
    }

    async for data in chain.astream_events(input_dict, version="v2", config=config):
        event_type = data["event"]
        node_name = data.get("metadata", {}).get("langgraph_node", "")
        # print(f"===>event_type: {event_type}, node_name: {node_name}")

        supported = False
        for event in SUPPORTED_EVENTS:
            if event_type in event and node_name in event[event_type]:
                supported = True
                break

        if supported:
            print(f"event_type: {event_type}, node_name: {node_name}")
            if os.environ.get("APP_ENV") == "dev":
                debug_data = copy.deepcopy(data)

                if "messages" in debug_data.get("data", {}).get("output", {}):
                    debug_data["data"]["output"]["messages"] = []

                if "messages" in debug_data.get("data", {}).get("input", {}):
                    debug_data["data"]["input"]["messages"] = []

                if "context" in debug_data.get("data", {}).get("input", {}):
                    debug_data["data"]["input"]["context"] = []

                pprint.pprint(debug_data, indent=2, width=80, depth=None)
                print("\n---\n")

            yield json.dumps(data, default=default_serialization) + "\n"

    yield json.dumps(EndEvent(), default=default_serialization) + "\n"


# Routes
@app.get("/")
async def redirect_root_to_docs() -> RedirectResponse:
    """Redirect the root URL to the API documentation."""
    return RedirectResponse("/docs")


@app.post("/stream_events")
async def stream_chat_events(request: InputState) -> StreamingResponse:
    """Stream chat events in response to an input request."""
    return StreamingResponse(
        stream_event_response(request), media_type="text/event-stream"
    )


@app.get("/threads/{thread_id}")
async def thread_state(thread_id: str) -> list:
    """Get the current thread state."""

    config = {"configurable": {"thread_id": thread_id}}

    # Get the workflow type and response from the state
    output_state = await retrieval_graph.aget_state(config=config)

    return output_state


register_exception_handlers(app)

# Main execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
