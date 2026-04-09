import folder_paths
import os
from PIL import ImageFont

body_part_names = ["Head", "Neck", "Shoulder", "Torso", "RArm", "RForearm", "LArm", "LForearm"]
script_directory = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
folder_paths.add_model_folder_path("bboxnodes_fonts", os.path.join(script_directory, "fonts"))
font_path = folder_paths.get_full_path("bboxnodes_fonts", "FreeMono.ttf")
font = ImageFont.truetype(font_path, 20)

def push_bbox(x, y, width, height, draw_boxes):
    bbox = [x, y, width, height]
    draw_boxes.append(bbox)

# convert tracking box into internal box for further processing
def push_tracking(frame_index, person_index, part, tracking, dummy_box, draw_boxes):
    if tracking:
        width = int(tracking[4])
        height = int(tracking[5])
        x1 = max(int(tracking[0]), 0)
        y1 = max(int(tracking[1]), 0)
        x2 = min(int(tracking[2]), width-1)
        y2 = min(int(tracking[3]), height-1)
        bbox = [x1, y1, x2, y2, part]
        if draw_boxes.get(frame_index) is None:
            draw_boxes[frame_index] = {}
        if draw_boxes[frame_index].get(person_index) is None:
            draw_boxes[frame_index][person_index] = []
        if (0 <= x1 < x2 < width
                and 0 <= y1 < y2 < height
                and x2 - x1 < width - 1
                and y2 - y1 < height - 1):
            draw_boxes[frame_index][person_index].append(bbox)
        else:
            if dummy_box:
                draw_boxes[frame_index][person_index].append([0, 0, 1, 1, part])

# parse tracking data from InstanceDiffusion node
def parse_tracking(tracking, body_part, person_index, dummy_box, boxes):
    for body_part_name in body_part_names:
        if body_part_name == body_part or body_part == "All":
            body_part_data = tracking.get(body_part_name)
            if body_part_data:
                # filtering out body parts belonging to other persons
                for current_person in body_part_data:
                    # 0 for all persons, >=1 for a specific person
                    if person_index == 0 or current_person == person_index - 1:
                        current_person_body_part_data = body_part_data.get(current_person)
                        if current_person_body_part_data:
                            # iterate through frames belonging to current person
                            for frame_index in current_person_body_part_data:
                                tracking_box = current_person_body_part_data.get(frame_index)
                                push_tracking(frame_index,
                                              current_person,
                                              body_part_name,
                                              tracking_box,
                                              dummy_box,
                                              boxes)


# Drawing frame number in the top left corner of the frame
def draw_frame_index(frame_index, draw):
    text = f"{frame_index}"
    text_position = (0, 0)
    color = (255, 0, 0)
    draw.text(text_position, text, fill=color, font=font)

# parse schedule string
def parse_schedule(frames_schedule, schedule):
    if schedule:
        schedule_parts = schedule.split(",")
        for schedule_part in schedule_parts:
            if not ':' in schedule_part:
                continue
            schedule_frame_index, schedule_person_index = schedule_part.split(":")
            if schedule_frame_index and schedule_person_index:
                frames_schedule[int(schedule_frame_index.strip())] = int(schedule_person_index.strip())
