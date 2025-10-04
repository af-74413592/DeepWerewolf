import ray
from copy import deepcopy

# from agentlightning.instrumentation.vllm import instrument_vllm, ChatCompletionResponsePatched
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
# from vllm.entrypoints.openai.protocol import ChatCompletionRequest, ErrorResponse
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest, ErrorResponse
# from verl.workers.rollout.vllm_rollout.vllm_async_server import AsyncvLLMServer
from verl.workers.rollout.sglang_rollout.async_sglang_server import AsyncSGLangServer
import asyncio
def _unwrap_ray_remote(cls):
    if hasattr(cls, "__ray_actor_class__"):
        cls = cls.__ray_actor_class__
    return cls


# @ray.remote(num_cpus=1)
# class PatchedvLLMServer(_unwrap_ray_remote(AsyncvLLMServer)):

#     def __init__(self, *args, **kwargs):
#         instrument_vllm()
#         super().__init__(*args, **kwargs)

#         self.config = deepcopy(self.config)
#         self.config.rollout.multi_turn.tool_config_path = "/dev/null"

#     async def chat_completion(self, raw_request: Request):
#         """OpenAI-compatible HTTP endpoint.

#         API reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
#         """
#         request_json = await raw_request.json()
#         request = ChatCompletionRequest(**request_json)
#         generator = await self.openai_serving_chat.create_chat_completion(request, raw_request)

#         if isinstance(generator, ErrorResponse):
#             return JSONResponse(content=generator.model_dump(), status_code=generator.code)
#         if request.stream:
#             return StreamingResponse(content=generator, media_type="text/event-stream")
#         else:
#             return JSONResponse(content=generator.model_dump())

@ray.remote(num_cpus=1)
class PatchedSglangServer(_unwrap_ray_remote(AsyncSGLangServer)):

    def __init__(self, *args, **kwargs):
        # instrument_sglang()
        super().__init__(*args, **kwargs)

        self.config = deepcopy(self.config)
        self.config.rollout.multi_turn.tool_config_path = "/dev/null"

    async def chat_completion(self, raw_request: Request):
        """OpenAI-compatible HTTP endpoint.

        """
        request = await raw_request.json()
        # only send request to master worker in tp rank 0
        output_future = self.master_worker.chat_completion.remote(request)
        [outputs] = await asyncio.gather(output_future)
        return JSONResponse(content=outputs.model_dump())