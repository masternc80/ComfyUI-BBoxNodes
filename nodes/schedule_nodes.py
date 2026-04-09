from comfy_api.latest import io
from .utils import parse_tracking, body_part_names, parse_schedule


class BBoxScheduler(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="BBoxScheduler",
            display_name="BBoxes Scheduler",
            category="BBoxNodes",
            description = """
Filters out bounding boxes using person index and a scheduler (Optional)'
Not compatible with KJNodes bboxes
""",
            inputs=[
                io.BoundingBox.Input("bboxes",
                                     force_input=True,
                                     tooltip="Bounding boxes (not compatible with KJNodes)"
                                     ),
                io.Int.Input("person_index",
                                 min=0,
                                 max=10,
                                 step=1,
                                 default=0,
                                 optional=False,
                                 tooltip="The person index on the image starting from 1. 0 for all persons"
                                 ),
                io.Boolean.Input("insert_dummy_bbox",
                                 optional=False,
                                 tooltip="Insert dummy one-pixel bbox when all bboxes are filtered out"),
                io.String.Input("schedule",
                                optional=True,
                                tooltip="Schedule in format 'frame_index:person_index, ...'"),
            ],
            outputs=[
                io.BoundingBox.Output("bboxes",
                                tooltip="Filtered out bboxes (Not compatible with KJNodes)"),
            ],
        )


    @classmethod
    def execute(cls, bboxes, person_index, insert_dummy_bbox, schedule) -> io.NodeOutput:
        if not isinstance(bboxes, list):
            bboxes = [[bboxes]]
        elif len(bboxes) == 0:
            return io.NodeOutput(bboxes)

        result = [[]] * len(bboxes)
        frames_schedule = {}
        parse_schedule(frames_schedule, schedule)

        person_filter = 0
        for index, bboxes_frame in enumerate(bboxes):
            if frames_schedule and frames_schedule.get(index) is not None:
                person_filter = frames_schedule[index]

            if len(bboxes_frame) == 0:
                continue

            result_bbox = []

            for box_index, box in enumerate(bboxes_frame):
                if person_index == 0 or box_index == person_index - 1:
                    if person_filter == 0 or box_index == person_filter - 1:
                        result_bbox.append(box)

            if insert_dummy_bbox and result_bbox == []:
                result_bbox = [{"x": 0, "y": 0, "width": 1, "height": 1}]
            result[index] = result_bbox

        return io.NodeOutput(result)




class TrackingToBBoxScheduler(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TrackingToBBoxScheduler",
            display_name="Convert Tracking To BBoxes",
            category="BBoxNodes",
            description = """
Converts the provided tracking info (InstanceDiffusion) to BBoxes format. Can use filtering (Body part, person index) 
and a scheduler (Optional). Person index is 0 for all persons, or a specific person starting from 1
Compatible with InstanceDiffusion nodes. Not compatible with KJNodes bboxes
""",
            inputs=[
                io.Custom("TRACKING").Input("tracking",
                                     optional=False,
                                     tooltip="Tracking data from InstanceDiffusion"
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
                               options=["All"] + body_part_names,
                               default="All",
                               optional=False,
                               tooltip="The body part on the image to draw rectangle for"),
                io.Boolean.Input("insert_dummy_bbox",
                                 optional=False,
                                 tooltip="Insert dummy one-pixel bbox when all bboxes are filtered out"),
                io.String.Input("schedule",
                                optional=True,
                                tooltip="Schedule in format 'frame_index:person_index, ...'"),
            ],
            outputs=[
                io.BoundingBox.Output("bboxes",
                                tooltip="Output bboxes (Not compatible with KJNodes)"),
            ],
        )

    @classmethod
    def execute(cls, tracking, person_index, body_part, insert_dummy_bbox, schedule) -> io.NodeOutput:

        frames_schedule = {}
        parse_schedule(frames_schedule, schedule)

        tracking_boxes = {}
        # Iterate through body parts in the tracking data
        parse_tracking(tracking, body_part, person_index, True, tracking_boxes)

        person_filter = 0
        # initialize bboxes
        empty_bbox = {"x": 0, "y": 0, "width": 1, "height": 1}
        bboxes = [[]] * len(tracking_boxes)

        print(f"frames_schedule: {frames_schedule}")
        # go through selected tracking boxes
        for tracking_index in tracking_boxes:

            bboxes[tracking_index] = []
            persons_list = tracking_boxes[tracking_index]
            if frames_schedule and frames_schedule.get(tracking_index) is not None:
                person_filter = frames_schedule[tracking_index]

            for person in persons_list:
                if person_filter == 0 or person_filter == person + 1:
                    tracking_boxes_list = tracking_boxes[tracking_index][person]

                    for tracking_box in tracking_boxes_list:

                        x1, y1, x2, y2, person_body_part = tracking_box
                        # convert to bbox format
                        bbox = {"x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}

                        if bbox != empty_bbox:
                            if bboxes[tracking_index] != [empty_bbox]:
                                bboxes[tracking_index].append(bbox)
                            else:
                                bboxes[tracking_index] = [bbox]

            # replace empy bbox with dummy bbox if required
            if bboxes[tracking_index] == [] and insert_dummy_bbox:
                bboxes[tracking_index] = [empty_bbox]

        return io.NodeOutput(bboxes)
