from .nodes.image_nodes import *
from .nodes.schedule_nodes import *
from comfy_api.latest import ComfyExtension, io

class BboxNodesExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [BboxesVisualize, TrackingVisualize, TrackingToBBoxScheduler, BBoxScheduler]

async def comfy_entrypoint() -> ComfyExtension:
    return BboxNodesExtension()

