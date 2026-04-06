from comfy_api.latest import io
import torch


def push_bbox(x, y, width, height, draw_boxes):
    bbox = [x, y, width, height]
    draw_boxes.append(bbox)

class BboxesVisualize:

    def define_schema(self):
        return io.Schema(
            node_id="VisualizeBBoxes",
            category="BBoxNodes",
            search_aliases=["Visualize", "bbox", "bounding box"],
            description="Visualizes the provided bboxes on the image for specified person. Specify 0 for all persons",
            inputs=[
                io.Image.Input("image"),
                io.BoundingBox.Input("bboxes", force_input=True),
                io.Int.Input("line_width", default=0, min=0, max=10, step=1,
                             tooltip="Width of the line"),
                io.Int.Input("person_index", default=0, min=0, max=5, step=1,
                             tooltip="Index of the person. 0 is all persons"),
            ],
            outputs=[
                io.Image.Output(tooltip="Image with boxes drawn on the image"),
            ],
        )

    @classmethod
    def execute(cls, image, bboxes, line_width, person_index) -> io.NodeOutput:

        draw_boxes = []
        image_list = []
        total_frames = image.shape[0]
        img_h = image.shape[1]
        img_w = image.shape[2]

        if not isinstance(bboxes, list):
            bboxes = [[bboxes]]
        elif len(bboxes) == 0:
            return io.NodeOutput(image)

        for frame_idx in range(total_frames):
            frame_bboxes = bboxes[min(frame_idx, len(bboxes) - 1)]
            if not frame_bboxes:
                continue

            if person_index == 0:
                for b in frame_bboxes:
                    x = b["x"]
                    y = b["y"]
                    width = b["width"]
                    height = b["height"]
                    push_bbox(x, y, width, height, draw_boxes)
            else:
                index = 0
                if person_index > 0 and len(frame_bboxes) <= person_index:
                    index = person_index - 1

                bbox = frame_bboxes[index]
                x = bbox["x"]
                y = bbox["y"]
                width = bbox["width"]
                height = bbox["height"]
                push_bbox(x, y, width, height, draw_boxes)

        for bbox in draw_boxes:

            x1 = min(bbox[0], img_w - 1)
            y1 = min(bbox[1], img_h - 1)
            x2 = min(img_w, bbox[0] + bbox[2])
            y2 = min(img_h, bbox[1] + bbox[3])
            # Permute the image dimensions
            image = image.permute(2, 0, 1)

            # Clone the image to draw bounding boxes
            img_with_bbox = image.clone()

            # Define the color for the bbox, e.g., red
            color = torch.tensor([1, 0, 0], dtype=torch.float32)

            # Ensure color tensor matches the image channels
            if color.shape[0] != img_with_bbox.shape[0]:
                color = color.unsqueeze(1).expand(-1, line_width)

            # Draw lines for each side of the bbox with the specified line width
            for lw in range(line_width):
                # Top horizontal line
                if y1 + lw < img_h:
                    img_with_bbox[:, y1 + lw, x1:x2] = color[:, None]

                # Bottom horizontal line
                if y2 - lw < img_h:
                    img_with_bbox[:, y2 - lw, x1:x2] = color[:, None]

                # Left vertical line
                if x1 + lw < img_w:
                    img_with_bbox[:, y1:y2, x1 + lw] = color[:, None]

                # Right vertical line
                if x2 - lw < img_w:
                    img_with_bbox[:, y1:y2, x2 - lw] = color[:, None]

            # Permute the image dimensions back
            img_with_bbox = img_with_bbox.permute(1, 2, 0).unsqueeze(0)
            image_list.append(img_with_bbox)

        return io.NodeOutput(image_list)