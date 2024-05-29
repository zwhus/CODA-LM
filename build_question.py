from PIL import Image, ImageDraw
import os
import numpy as np
import json
import argparse

def box_xyxy_expand2square(box, w, h):
    if w == h:
        x1, y1, x2, y2 = box
        return [round(x1 / w, 2), round(y1 / w, 2), round(x2 / w, 2), round(y2 / w, 2)]
    if w > h:
        x1, y1, x2, y2 = box
        y1 += (w - h) // 2
        y2 += (w - h) // 2
        box = [round(x1 / w, 2), round(y1 / w, 2), round(x2 / w, 2), round(y2 / w, 2)]
        return box
    assert w < h
    x1, y1, x2, y2 = box
    x1 += (h - w) // 2
    x2 += (h - w) // 2
    box = [round(x1 / h, 2), round(y1 / h, 2), round(x2 / h, 2), round(y2 / h, 2)]
    return box


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    
def load_gt_grouding(ref, image_folder, save_path, visualize_dir, box_width=2):
 
    gt_list = sorted(os.listdir(ref))
    os.makedirs(visualize_dir, exist_ok=True)
    data = []
    cnt = 0
    for json_name in gt_list:
        img_name = json_name.split('.')[0] + '.jpg'
        img = Image.open(os.path.join(image_folder, img_name))
        height, width, _ = np.array(img).shape
        # new_img = expand2square(img, tuple(
        #                 int(x * 255) for x in [0.48145466, 0.4578275, 0.40821073]))
        
        # image annotation
        with open(os.path.join(ref, json_name), 'r') as f:
            json_data = json.load(f)
        
        regional_data = json_data["region_perception"]
    
        for key, value in regional_data.items():
            # draw = ImageDraw.Draw(new_img)
            # new_h, new_w, _ = np.array(new_img).shape
            value['box'][2] = value['box'][0] + value['box'][2]
            value['box'][3] = value['box'][1] + value['box'][3]
            bbox = box_xyxy_expand2square(value['box'], width, height)
            
            # rect = [int(bbox[0] * new_w), int(bbox[1] * new_h), int(bbox[2] * new_w), int(bbox[3] * new_h)]
            # draw.rectangle(rect, outline="red", width=box_width)
            # output_path = os.path.join(visualize_dir, f"{json_name.split('.')[0]}_object_{key}.jpg")
            # new_img.save(output_path)
            data.append({
                "question_id": cnt,
                "image": img_name,
                "text": f"Please provide a description for this object and explain why this object affect ego car driving: <{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}>.",
                "label_name": value["category_name"]})
            cnt += 1

    # save question list
    with open(save_path, 'w') as file:
        for entry in data:
            json_str = json.dumps(entry)
            file.write(json_str + '\n')
    
def load_gt_visualize(ref, image_folder, save_path, visualize_dir, box_width=2):
 
    gt_list = sorted(os.listdir(ref))

    data = []
    cnt = 0
    os.makedirs(visualize_dir, exist_ok=True)
    for json_name in gt_list:
        # image info
        img_name = json_name.split('.')[0] + '.jpg'

        # image annotation
        with open(os.path.join(ref, json_name), 'r') as f:
            json_data = json.load(f)
        
        regional_data = json_data["region_perception"]
        for key, value in regional_data.items():
            img = Image.open(os.path.join(image_folder, img_name))
            draw = ImageDraw.Draw(img)
            rect = [value['box'][0], value['box'][1], value['box'][0] + value['box'][2], value['box'][1] + value['box'][3]]
            # x1, y1, x2, y2
            draw.rectangle(rect, outline="red", width=box_width)
            output_path = os.path.join(visualize_dir, f"{json_name.split('.')[0]}_object_{key}.jpg")
            img.save(output_path)
            data.append({
                "question_id": cnt,
                "image": output_path,
                "text": f"Please describe the object inside the red rectangle in the image and explain why it affect ego car driving.",
                "label_name": value["category_name"]})
            cnt += 1

    # save question list
    with open(save_path, 'w') as file:
        for entry in data:
            json_str = json.dumps(entry)
            file.write(json_str + '\n')

def build_stage1_stage2(args):
    ref_list = sorted(os.listdir(args.ref))
    stage1_data, stage2_data = [], []
    cnt = 0
    for json_name in ref_list:
        img_name = json_name.split('.')[0] + '.jpg'
        stage1_data.append({
                "question_id": cnt,
                "image": img_name,
                "text": "There is an image of traffic captured from the perspective of the ego car. Please focus on objects that have a great influence on ego car driving behavior in the scene, describe these objects and the reasons why they influence ego car driving."})
        stage2_data.append({
                "question_id": cnt,
                "image": img_name,
                "text": "There is an image of traffic captured from the perspective of the ego car. Please provide driving suggestions for the ego car based on the current scene."})
        cnt += 1
    
    with open(args.stage1_question_file, 'w') as file:
        for entry in stage1_data:
            json_str = json.dumps(entry)
            file.write(json_str + '\n')

    with open(args.stage2_question_file, 'w') as file:
        for entry in stage2_data:
            json_str = json.dumps(entry)
            file.write(json_str + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/codalm_data/codalm_images/images")
    parser.add_argument("--visualize_dir", type=str, default="Mini/visualize")
    parser.add_argument("--ref", type=str, default="CODA-LM_v1.2key_sugg/Mini")
    parser.add_argument("--stage1_question_file", type=str, default="Mini/stage1_question.jsonl")
    parser.add_argument("--stage2_question_file", type=str, default="Mini/stage2_question.jsonl")
    parser.add_argument("--stage3_vis_question_file", type=str, default="Mini/stage3_vis_question.jsonl")
    parser.add_argument("--stage3_grouding_question_file", type=str, default="Mini/stage3_grouding_question.jsonl")
    args = parser.parse_args()

    load_gt_visualize(args.ref, args.image_folder, args.stage3_vis_question_file, visualize_dir=args.visualize_dir)
    build_stage1_stage2(args)
    load_gt_grouding(args.ref, args.image_folder, args.stage3_grouding_question_file, visualize_dir=args.visualize_dir)


