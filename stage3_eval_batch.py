from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, wait
from functools import partial
from tqdm import tqdm
import time
import os
import json
import argparse

class RegionEval(object):

    def __init__(self, reference_path, prediction_path):
        self.system_prompt = "You are an impartial judge tasked with evaluating text similarity and relevance of the reference text and autonomous driving AI assistant's predicted text. Be as objective as possible. Do not allow the length of the predicted text to influence your \
                    evaluation. After providing your short explanation, you must rate on a scale from 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[10]]\"."
        self.gt_data = defaultdict(list)
        self.predict_data = defaultdict(list)
        self.load_gt(reference_path)
        self.load_predict(prediction_path)
    
    def load_predict(self, prediction_path):
        with open(prediction_path, 'r') as file:
            for line in file:
                line_data = json.loads(line.strip())
                new_label_name = self.convert_label(line_data["label_name"])
                self.predict_data[new_label_name].append({
                    'prompt': line_data['prompt'],
                    'caption': line_data['text']
                })
    
    def load_gt(self, reference_path):
        gt_list = sorted(os.listdir(reference_path))
        for json_name in gt_list:
            with open(os.path.join(reference_path, json_name), 'r') as f:
                json_data = json.load(f)
            for object_id, object_data in json_data.items():
                new_label_name = self.convert_label(object_data['category_name'])

                self.gt_data[new_label_name].append({
                    'image_name': f"{json_name.split('.')[0]}_bbox_{object_id}.jpg",
                    'box_type': 'xywh', # top xy
                    'bbox': object_data['box'],
                    'caption': object_data['description and explanation']
                })
            
    def convert_label(self, category_name):
        """
        {
            "vehicle": {"car", "truck", "tram", "tricycle",, "bus", "trailer", "construction_vehicle", "recreational_vehicle"},
            "vru": {"pedestrian", "cyclist", "bicycle", "moped", "motorcycle", "stroller", "wheelchair", "cart"},
            "traffic sign": {"warning_sign", "traffic_sign"},
            "traffic light": {"traffic_light"},
            "traffic cone": {"traffic_cone"},
            "traffic_facility", {"barrier", "bollard"},
            "others": {"dog", "cat", "sentry_box", "traffic_box", "traffic_island", "debris", "suitcace", "dustbin",
                        "concrete_block", "machinery", "chair", "phone_booth", "basket", "cardboard", "carton", "garbage", "garbage_bag",
                        "plastic_bag", "stone", "tire", "misc"}
        }
        {
            "vehicle": {2, 3, 4, 5, 6, 16, 14, 15}
            "vru": {0, 1, 7, 8, 9, 10, 11, 12}},
            "traffic sign": {20, 26},
            "traffic light": {25},
            "traffic cone": {23},
            "traffic_facility", {18, 19},
            "miscellaneous": {16, 17, 21, 22, 24, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42}
        }
        """

        label_dict = {
            "vehicle": ["car", "truck", "tram", "tricycle","bus", "trailer", "construction_vehicle", "recreational_vehicle"],
            "vru": ["pedestrian", "cyclist", "bicycle", "moped", "motorcycle", "stroller", "wheelchair", "cart"],
            "traffic_sign": ["warning_sign", "traffic_sign"],
            "traffic_light": ["traffic_light"],
            "traffic_cone": ["traffic_cone"], 
            "barrier": ["barrier", "bollard"],
            "miscellaneous": ["dog", "cat", "sentry_box", "traffic_box", "traffic_island", "debris", "suitcace", "dustbin", "concrete_block", "machinery", "chair", "phone_booth", "basket", "cardboard", "carton", "garbage", "garbage_bag", "plastic_bag", "stone", "tire", "misc"],
        }
        self.label_info = {
            "vehicle": 0,
            "vru": 1,
            "traffic_sign": 2,
            "traffic_light": 3,
            "traffic_cone": 4,
            "barrier": 5,
            "miscellaneous": 6
        }
        for new_name, label_info in label_dict.items():
            if category_name in label_info:
                return new_name
    
    def get_class(self):
        return self.label_info
    
    def create_messages(self, message):
        ret = []
        # system prompt
        ret.append({
            "role": "system",
            "content": self.system_prompt
        })

        # few shot example
        template = "[The Start of Reference Text]\n{}\n[The End of Reference Text]\n\n[The Start of Prediction Text]\n{}\n[The End of Prediction Text]"

        ret.append({
            "role": "user", 
            "content": template.format(message["reference"], message["prediction"])
        })

        return ret
    
    def get_class_messages(self, label_name):
        results = []
        txt_names = []
        gt, pred = self.gt_data[label_name], self.predict_data[label_name]  # List[Dict]
        for index in tqdm(range(len(gt))):
            hypo = pred[index]
            ref = gt[index]
            message = dict()
            message["prediction"] = hypo["caption"]
            message["reference"] = ref["caption"]
            results.append(self.create_messages(message))
            txt_names.append(f"{ref['image_name'].split('.')[0]}.txt")
            
        return results, txt_names

class GPTBatcher:
    """
    Copy from https://github.com/fengsxy/gpt_batch

    Parameters:
        api_key (str): API key for authenticating requests to the OpenAI API.
        model_name (str, optional): Specifies the GPT model version. Default is 'gpt-3.5-turbo-0125'.
        system_prompt (str, optional): Initial text or question to seed the model with. Default is empty.
        temperature (float, optional): Sets the creativity of the responses. Default is 1.
        num_workers (int, optional): Number of parallel workers for request handling. Default is 64.
        timeout_duration (int, optional): Timeout for API responses in seconds. Default is 60.
        retry_attempts (int, optional): How many times to retry a failed request. Default is 2.
    """

    def __init__(self, 
                 api_key, 
                 model_name="gpt-3.5-turbo-0125", 
                 system_prompt="", 
                 temperature=0,
                 num_workers=64,
                 timeout_duration=60,
                 retry_attempts=2,
                 api_base_url=None):
        
        self.client = OpenAI(api_key=api_key, base_url = api_base_url)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.num_workers = num_workers
        self.timeout_duration = timeout_duration
        self.retry_attempts = retry_attempts
        self.miss_index =[]
        if api_base_url:
            self.client.base_url = api_base_url

    def get_attitude(self, ask_text):
        index, ask_text = ask_text
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=ask_text,
                temperature=self.temperature,
            )
            return (index, completion.choices[0].message.content)
        except Exception as e:
            print(f"Error occurred: {e}")
            self.miss_index.append(index)
            return (index, None)

    def process_attitude(self, message_list):
        new_list = []
        num_workers = self.num_workers
        timeout_duration = self.timeout_duration
        retry_attempts = 2
    
        executor = ThreadPoolExecutor(max_workers=num_workers)
        message_chunks = list(self.chunk_list(message_list, num_workers))
        try:
            for chunk in tqdm(message_chunks, desc="Processing messages"):
                future_to_message = {executor.submit(self.get_attitude, message): message for message in chunk}
                for _ in range(retry_attempts):
                    done, not_done = wait(future_to_message.keys(), timeout=timeout_duration)
                    for future in not_done:
                        future.cancel()
                    new_list.extend(future.result() for future in done if future.done())
                    if len(not_done) == 0:
                        break
                    future_to_message = {executor.submit(self.get_attitude, future_to_message[future]): future for future in not_done}
        except Exception as e:
            print(f"Error occurred: {e}")
        finally:
            executor.shutdown(wait=False)
            return new_list

    def complete_attitude_list(self,attitude_list, max_length):
        completed_list = []
        current_index = 0
        for item in attitude_list:
            index, value = item
            # Fill in missing indices
            while current_index < index:
                completed_list.append((current_index, None))
                current_index += 1
            # Add the current element from the list
            completed_list.append(item)
            current_index = index + 1
        while current_index < max_length:
            print("Filling in missing index", current_index)
            self.miss_index.append(current_index)
            completed_list.append((current_index, None))
            current_index += 1
        return completed_list

    def chunk_list(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def handle_message_list(self, message_list):
        indexed_list = [(index, data) for index, data in enumerate(message_list)]
        max_length = len(indexed_list)
        attitude_list = self.process_attitude(indexed_list)
        attitude_list.sort(key=lambda x: x[0])
        attitude_list = self.complete_attitude_list(attitude_list, max_length)
        attitude_list = [x[1] for x in attitude_list]
        return attitude_list
    
    def get_miss_index(self):
        return self.miss_index


if __name__ == "__main__":
  
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_path", type=str, default="")
    parser.add_argument("--prediction_path", type=str, default="")
    parser.add_argument("--save_path", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--model_name", type=str, default="gpt-4-1106-preview")
    parser.add_argument("--api_key", type=str, default="sk-ZviaVZ9N7MOtjVL8EbA43e034cFf4b8e83902f407630FcBf")
    parser.add_argument("--api_base_url", type=str, default="https://api.gptplus5.com/v1")
    
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "gpt_result"), exist_ok=True)
    region_eval = RegionEval(args.reference_path, args.prediction_path)
    batcher = GPTBatcher(
        api_key=args.api_key, 
        model_name=args.model_name, 
        num_workers=args.num_workers,
        api_base_url=args.api_base_url)
    
    label_info = region_eval.get_class()
    for label_name in tqdm(self.label_info.keys()):
        cls_score = []
        rets, txt_names = get_class_messages(label_name)
        results = batcher.handle_message_list(rets)
        
        for idx, txt_name in tqdm(enumerate(txt_names)):
            output = results[idx]
            if output == None:
                continue
                print(f"Missing {txt_name} output")
                
            try:
                cls_score.append(int(output.split("Rating: [[")[1].split("]]")[0]))
            except:
                try:
                    cls_score.append(int(output.split("rating is: [[")[1].split("]]")[0]))
                except:
                    try:
                        cls_score.append(int(output.split("[[")[1].split("]]")[0]))
                    except:
                        print(f"Missing extract score from {txt_name}")
                        
            with open(os.path.join(args.save_path, "gpt_result", txt_name), "w") as f:
                f.write(output)
        
        with open(os.path.join(self.save_path, f"{label_name}.txt"), "w") as fp:
            fp.write(f"computing gpt score: {sum(cls_score) / len(cls_score)}\n")
        print(f"Label: {label_name}, GPT-Score: {sum(cls_score) / len(cls_score)}")
        
   
