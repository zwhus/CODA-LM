from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, wait
from functools import partial
from tqdm import tqdm
from openai import OpenAI
import time
import os
import json
from tqdm import tqdm
import argparse

class GPTBatcher:
    """
    A class to handle batching and sending requests to the OpenAI GPT model efficiently.

    Attributes:
        client (OpenAI): The client instance to communicate with the OpenAI API using the provided API key.
        model_name (str): The name of the GPT model to be used. Default is 'gpt-3.5-turbo-0125'.
        system_prompt (str): Initial prompt or context to be used with the model. Default is an empty string.
        temperature (float): Controls the randomness of the model's responses. Higher values lead to more diverse outputs. Default is 1.
        num_workers (int): Number of worker threads used for handling concurrent requests. Default is 64.
        timeout_duration (int): Maximum time (in seconds) to wait for a response from the API before timing out. Default is 60 seconds.
        retry_attempts (int): Number of retries if a request fails. Default is 2.
        miss_index (list): Tracks the indices of requests that failed to process correctly.

    Parameters:
        api_key (str): API key for authenticating requests to the OpenAI API.
        model_name (str, optional): Specifies the GPT model version. Default is 'gpt-3.5-turbo-0125'.
        system_prompt (str, optional): Initial text or question to seed the model with. Default is empty.
        temperature (float, optional): Sets the creativity of the responses. Default is 1.
        num_workers (int, optional): Number of parallel workers for request handling. Default is 64.
        timeout_duration (int, optional): Timeout for API responses in seconds. Default is 60.
        retry_attempts (int, optional): How many times to retry a failed request. Default is 2.
    """

    def __init__(self, api_key, model_name="gpt-3.5-turbo-0125", system_prompt="",temperature=0,num_workers=64,timeout_duration=60,retry_attempts=2,api_base_url=None):
        
        self.client = OpenAI(api_key=api_key, base_url = api_base_url)
        self.model_name = model_name
        self.system_prompt = "You are an impartial judge tasked with evaluating the quality of predicted text provided by autonomous driving AI assistant. You will compare this prediction text to a reference text, focusing on the description of objects that influence the driving behavior of ego car, and the explanation of why these objects impact. Your evaluation criteria should include accuracy(checking \
                            if the predicted text correctly identifies objects mentioned the reference text), suppression hallucination(ensuring that objects not mentioned in the reference text are not erroneously included in the predicted text), correlation(sessing if the reasons for the objects' impact on the ego car's driving behavior are consistent between the reference and predicted text). Be as objective as possible. \
                            Do not allow the length of the predicted text to influence your evaluation. Maximize your text comprehension capabilities to freely match objects with high similarity, appropriately ignoring the relative positions and color attributes of the objects. After providing your short explanation, you must rate the response on a scale from 1 to 10 by strictly following this format: \"[[rating]]\", for example: \"Rating: [[10]]\"."
        self.temperature = temperature
        self.num_workers = num_workers
        self.timeout_duration = timeout_duration
        self.retry_attempts = retry_attempts
        self.miss_index =[]
        if api_base_url:
            self.client.base_url = api_base_url


    def create_messages(self, message):
        ret = []
        # system prompt
        ret.append({
            "role": "system",
            "content": self.system_prompt
        })

        # few shot example
        few_shot = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene_few_shot")
        with open(os.path.join(few_shot, "high.json")) as f:
            high_data = json.load(f)
        with open(os.path.join(few_shot, "low.json")) as f:
            low_data = json.load(f)

        template = "[The Start of Reference Text]\n{}\n[The End of Reference Text]\n\n[The Start of Prediction Text]\n{}\n[The End of Prediction Text]"

        # high example
        ret.append({
            "role": "user", 
            "content": template.format(high_data["reference"], high_data["prediction"])
        })
        ret.append({
            "role": "assistant", 
            "content": high_data["response"]
        })

        # low example
        ret.append({
            "role": "user", 
            "content": template.format(low_data["reference"], low_data["prediction"])
        })
        ret.append({
            "role": "assistant", 
            "content": low_data["response"]
        })

        ret.append({
            "role": "user", 
            "content": template.format(message["reference"], message["prediction"])
        })
        return ret

    def get_attitude(self, ask_text):
        index, ask_text = ask_text
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=ask_text,
                temperature=self.temperature,
            )
            # txt_name = json_name.replace(".json", ".txt")
            # with open(os.path.join(args.save_path, txt_name), "w") as f:
            #     f.write(output)
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

    def handle_message_list(self,message_list):
        indexed_list = [(index, data) for index, data in enumerate(message_list)]
        max_length = len(indexed_list)
        attitude_list = self.process_attitude(indexed_list)
        attitude_list.sort(key=lambda x: x[0])
        attitude_list = self.complete_attitude_list(attitude_list, max_length)
        attitude_list = [x[1] for x in attitude_list]
        return attitude_list
    
    def process_embedding(self,message_list):
            new_list = []
            executor = ThreadPoolExecutor(max_workers=self.num_workers)
            # Split message_list into chunks
            message_chunks = list(self.chunk_list(message_list, self.num_workers))
            fixed_get_embedding = partial(self.get_embedding)
            for chunk in tqdm(message_chunks, desc="Processing messages"):
                future_to_message = {executor.submit(fixed_get_embedding, message): message for message in chunk}
                for i in range(self.retry_attempts):
                    done, not_done = wait(future_to_message.keys(), timeout=self.timeout_duration)
                    for future in not_done:
                        future.cancel()
                    new_list.extend(future.result() for future in done if future.done())
                    if len(not_done) == 0:
                        break
                    future_to_message = {executor.submit(fixed_get_embedding, future_to_message[future]): future_to_message[future] for future in not_done}
            executor.shutdown(wait=False)
            return new_list
    def get_embedding(self,text):
        index,text = text
        response = self.client.embeddings.create(
        input=text,
        model=self.model_name)
        return (index,response.data[0].embedding)

    def handle_embedding_list(self,message_list):
        indexed_list = [(index, data) for index, data in enumerate(message_list)]
        max_length = len(indexed_list)
        attitude_list = self.process_embedding(indexed_list)
        attitude_list.sort(key=lambda x: x[0])
        attitude_list = self.complete_attitude_list(attitude_list, max_length)
        attitude_list = [x[1] for x in attitude_list]
        return attitude_list
    
    def get_miss_index(self):
        return self.miss_index

    # Add other necessary methods similar to the above, refactored to fit within this class structure.


if __name__ == "__main__":
    # LLava-1.5: 3.167  ---- 3.3  ----- 3.2
    # Gemini-pro: 2.467 ---- 2.8 ---- 2.6
    # MiniGPTV2: 2.6333 
    # shikra: 2.3
    # ShareGPT4V: 3.167 ---- 3.1
    # GPT4V: 4.3  ------ 4.8 ---- 4.6
    # Qwen: 2.466

    # stage1 cost: 30 image ---- 14 (2280 * 20 = 45600)
    # Todo: multi prograss
    parser = argparse.ArgumentParser()
    # /19969306569/GPT4V_Data/ann_afrefine/final_annotations_stage1_1-154_com
    # /19969306569/huawei_annotation/final_ann_stage12_4_8/final_annotations
    # /19969306569/huawei_annotation/eval/stage1_shikra_eval
    parser.add_argument("--reference_path", type=str, default="/19969306569/huawei_annotation/final_ann_stage12_4_8/final_annotations")
    parser.add_argument("--prediction_path", type=str, default="/19969306569/XTuner/eval_res/codalm_llava_stage1_result.jsonl")
    parser.add_argument("--save_path", type=str, default="/19969306569/XTuner/eval/stage1_llava_vicuna_eval_test")
    args = parser.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    json_list = sorted(os.listdir(args.reference_path))
    answers = [json.loads(q) for q in open(os.path.expanduser(args.prediction_path), "r")]
    batcher = GPTBatcher(api_key='sk-ZviaVZ9N7MOtjVL8EbA43e034cFf4b8e83902f407630FcBf', model_name='gpt-4-1106-preview', api_base_url='https://api.gptplus5.com/v1')
    score = []
    rets = []
    for idx, json_name in tqdm(enumerate(json_list)):
        message= dict()
        message["prediction"] = answers[idx]['answer']
        
        with open(os.path.join(args.reference_path, json_name), "r") as f:
            data = json.load(f)
        info = []
        for key, value in data.items():
            if key == "suggestions": 
                continue
            if isinstance(value, list):
                for v in value:
                    info.append(v["description"])
                    info.append(v["explanation"])
            else:
                info.append(v["description"])
                info.append(v["explanation"])
        
        message["reference"] = " ".join(info)
        ret = batcher.create_messages(message)
        rets.append(ret)
    results = batcher.handle_message_list(rets)
    for idx, json_name in tqdm(enumerate(json_list)):
        output = results[idx]
        # output = results[idx].choices[0].message.content
        txt_name = json_name.replace(".json", ".txt")
        with open(os.path.join(args.save_path, txt_name), "w") as f:
            f.write(output)
    
    # cal score
    all_score = []
    for name in sorted(os.listdir(args.save_path)):
        with open(os.path.join(args.save_path, name)) as f:
            output = f.read()
            # import pdb; pdb.set_trace()
            try:
                all_score.append(int(output.split("Rating: [[")[1].split("]]")[0]))
            except:
                try:
                    all_score.append(int(output.split("rating is: [[")[1].split("]]")[0]))
                except:
                    try:
                        all_score.append(int(output.split("[[")[1].split("]]")[0]))
                    except:
                        print(f"name: {name}, {output}")

    print(f"all_score: {sum(all_score)/len(all_score)}")