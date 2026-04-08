import torch
from utils import push_bbox, parse_tracking, body_part_names, draw_frame_index
from torchvision import transforms
from comfy.utils import ProgressBar
from comfy_api.latest import io
from PIL import ImageDraw


colors = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0.5, 0.5, 0],
    [0, 0.5, 0.5],
    [0.5, 0, 0.5],
    [0.25, 0.75, 0],
    [0.75, 0.25, 0],
    [0, 0.25, 0.75],
    [0, 0.75, 0.25]
]

class BboxesVisualize(io.ComfyNode):


    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BboxesVisualize",
            display_name="Bounding Boxes Visualize",
            category="BBoxNodes",
            description = """
    Visualizes the provided bboxes on the image for specified person(s). Specify 0 for all persons.
    Not compatible with KJNodes bboxes
    """,
            inputs=[
                io.Image.Input("images",
                               optional=False,
                               tooltip="The input images to process"
                               ),
                io.BoundingBox.Input("bboxes",
                                     force_input=True,
                                     tooltip="Bounding boxes (not compatible with KJNodes)"
                                     ),
                io.Int.Input("line_width",
                                 min=1,
                                 max=5,
                                 step=1,
                                 default=1,
                                 optional=False,
                                 tooltip="Rectangle border width"
                                 ),
                io.Int.Input("person_index",
                                 min=0,
                                 max=10,
                                 step=1,
                                 default=0,
                                 optional=False,
                                 tooltip="The person index on the image starting from 1. 0 for all persons"
                                 ),
            ],
            outputs=[
                io.Image.Output("images",
                                tooltip="Output images with drawn bounding boxes"),
            ],
        )

    @classmethod
    def execute(cls, images, bboxes, line_width, person_index) -> io.NodeOutput:

        image_list = []
        if images.shape[0] == 0:
            raise ValueError("Input image batch is empty")

        if not isinstance(bboxes, list):
            bboxes = [[bboxes]]
        elif len(bboxes) == 0:
            return io.NodeOutput(images)

        steps = images.shape[0]
        pbar = ProgressBar(steps)
        batch_count = 0

        for image, frame_bboxes in zip(images, bboxes):

            draw_boxes = []
            for index, box in enumerate(frame_bboxes):
                if person_index == 0 or index == person_index - 1:
                    x = box["x"]
                    y = box["y"]
                    width = box["width"]
                    height = box["height"]
                    push_bbox(x, y, width, height, draw_boxes)

            # Permute the image dimensions
            img_with_bbox = image.permute(2, 0, 1)
            pil_image = transforms.ToPILImage()(img_with_bbox)
            draw = ImageDraw.Draw(pil_image)

            # go through selected draw boxes
            for color_index, bbox in enumerate(draw_boxes):

                # Define the color for the bbox, e.g., red
                color = tuple(int(255 * x) for x in colors[color_index])[:3]

                # Draw the bbox rectangle
                x, y, width, height = bbox
                draw.rectangle([x, y, x + width, y + width], outline=color, width=line_width)

            #Drawing frame number in the top left corner of the frame
            draw_frame_index(batch_count, draw)

            # Permute the image dimensions back
            img_with_bbox = transforms.ToTensor()(pil_image).permute(1, 2, 0)
            image_list.append(img_with_bbox)
            pbar.update(batch_count)
            batch_count += 1

        return io.NodeOutput(torch.stack(image_list).cpu().float())

class TrackingVisualize(io.ComfyNode):


    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TrackingVisualize",
            display_name="Tracking Data Visualize",
            category="BBoxNodes",
            description = """
    Visualizes the provided tracking info (InstanceDiffusion) on the image for all or specified person.
    Similar to KJNode 'DrawInstanceDiffusionTracking'. Compatible with InstanceDiffusion nodes
    """,
            inputs=[
                io.Image.Input("images",
                               optional=False,
                               tooltip="The input images to process"
                               ),
                io.Custom("TRACKING").Input("tracking",
                                     optional=False,
                                     tooltip="Tracking data from InstanceDiffusion"
                                     ),
                io.Int.Input("line_width",
                                 min=1,
                                 max=5,
                                 step=1,
                                 default=1,
                                 optional=False,
                                 tooltip="Rectangle border width"
                                 ),
                io.Int.Input("person_index",
                                 min=0,
                                 max=10,
                                 step=1,
                                 default=0,
                                 optional=False,
                                 tooltip="The person index on the image starting from 1. 0 for all persons"
                                 ),
                io.Combo.Input("body_part",
                               options=["All", "Head", "Neck", "Shoulder", "LArm", "RArm", "LForearm", "RForearm", "Torso"],
                               default="All",
                               optional=False,
                               tooltip="The body part on the image to draw rectangle for")
            ],
            outputs=[
                io.Image.Output("images",
                                tooltip="Output images with drawn bounding boxes"),
            ],
        )

    @classmethod
    def execute(cls, images, tracking, line_width, person_index, body_part) -> io.NodeOutput:

        image_list = []

        steps = images.shape[0]
        pbar = ProgressBar(steps)

        draw_boxes = {}
        # Iterate through body parts in the tracking data
        parse_tracking(tracking, body_part, person_index, False, draw_boxes)

        for frame_index,image in enumerate(images):

            # Permute the image dimensions
            img_with_bbox = image.permute(2, 0, 1)
            pil_image = transforms.ToPILImage()(img_with_bbox)
            draw = ImageDraw.Draw(pil_image)

            if draw_boxes.get(frame_index):
                bboxes = draw_boxes[frame_index]

                # Draw collected boxes
                for bbox in bboxes:

                    x1, y1, x2, y2, part = bbox
                    color_index = body_part_names.index(part)

                    # Define the color for the bbox, e.g., red
                    color = tuple(int(255 * x) for x in colors[color_index])[:3]

                    # Draw the tracking box rectangle
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

            #Drawing frame number in the top left corner of the frame
            draw_frame_index(frame_index, draw)

            # Permute the image dimensions back
            img_with_bbox = transforms.ToTensor()(pil_image).permute(1, 2, 0)
            image_list.append(img_with_bbox)
            pbar.update(frame_index)

        return io.NodeOutput(torch.stack(image_list).cpu().float())
