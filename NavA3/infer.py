import os
import json
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
import torch.distributed as dist

#os.environ["MASTER_ADDR"] = "localhost"
#os.environ["MASTER_PORT"] = "29500"
#dist.init_process_group(backend="nccl", rank=0, world_size=1)  # 单卡配置


def extract_coordinate_data(response_text, canvas_w=640, canvas_h=480):
    import re
    coordinate_pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    found_matches = re.findall(coordinate_pattern, response_text)
    coordinate_list = []
    pixel_coordinates = []
    
    for match_item in found_matches:
        coord_values = [float(val) if '.' in val else int(val) for val in match_item.split(',')]
        if len(coord_values) == 2:
            pos_x, pos_y = coord_values
            coordinate_list.append(('point', (pos_x, pos_y)))
            pixel_coordinates.append((pos_x, pos_y))
        elif len(coord_values) == 4:
            left_x, top_y, right_x, bottom_y = coord_values
            coordinate_list.append(('rect', (left_x, top_y, right_x, bottom_y)))
            x_range = np.linspace(left_x, right_x, num=10, dtype=int)
            y_range = np.linspace(top_y, bottom_y, num=10, dtype=int)
            grid_x, grid_y = np.meshgrid(x_range, y_range)
            pixel_coordinates.extend(list(np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)))
    
    return coordinate_list, np.array(pixel_coordinates) if pixel_coordinates else np.empty((0, 2))


def create_visualization(img_file, coord_data, save_path, prompt_text=None):
    source_img = Image.open(img_file)
    img_w, img_h = source_img.size
    
    figure, axis = plt.subplots(1, figsize=(10, 6))
    plt.imshow(source_img)
    
    highlight_color = 'cyan'
    border_thickness = 2
    
    for idx, (shape_type, position_data) in enumerate(coord_data):
        if shape_type == 'point':
            center_x, center_y = position_data
            point_marker = patches.Circle((center_x, center_y), radius=8, 
                                        edgecolor=highlight_color, facecolor=highlight_color, 
                                        linewidth=border_thickness)
            axis.add_patch(point_marker)
        elif shape_type == 'rect':
            start_x, start_y, end_x, end_y = position_data
            rectangle = patches.Rectangle((start_x, start_y), end_x-start_x, end_y-start_y, 
                                        linewidth=border_thickness, edgecolor=highlight_color, 
                                        facecolor='none')
            axis.add_patch(rectangle)
    
    plt.axis('off')
    if prompt_text:
        display_title = prompt_text[:100] + '...' if len(prompt_text) > 100 else prompt_text
        plt.title(display_title, fontsize=10, pad=10)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()


def initialize_qwen_model(model_directory):
    from qwen_vl_utils import process_vision_info
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    
    vision_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_directory,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",  # 性能优化
        trust_remote_code=True
    )
    vision_model.eval()
    
    text_processor = AutoProcessor.from_pretrained(model_directory, trust_remote_code=True)
    return {"vision_model": vision_model, "text_processor": text_processor}


def execute_inference(query_text, img_path, model_components):
    from qwen_vl_utils import process_vision_info
    from PIL import Image
    import torch

    model_instance = model_components["vision_model"]
    processor_instance = model_components["text_processor"]

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_path},
                {"type": "text", "text": query_text},
            ],
        }
    ]

    formatted_text = processor_instance.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    img_data, vid_data = process_vision_info(conversation)

    model_inputs = processor_instance(
        text=[formatted_text],
        images=img_data,
        videos=vid_data,
        padding=True,
        return_tensors="pt",
    )
    model_inputs = model_inputs.to(model_instance.device)

    with torch.no_grad():
        output_tokens = model_instance.generate(**model_inputs, max_new_tokens=128)

    trimmed_tokens = [
        out_seq[len(in_seq):] for in_seq, out_seq in zip(model_inputs.input_ids, output_tokens)
    ]
    decoded_response = processor_instance.batch_decode(
        trimmed_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return decoded_response[0].strip()


def run_batch_inference():
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument('--model-path', type=str, required=True, help='model path')
    cmd_parser.add_argument('--input-json', type=str, default='demo_input/instructions.json', help='instruction json')
    cmd_parser.add_argument('--image-dir', type=str, default='demo_input', help='image')
    cmd_parser.add_argument('--output-dir', type=str, default='demo_output', help='result')
    cmd_args = cmd_parser.parse_args()

    os.makedirs(cmd_args.output_dir, exist_ok=True)
    
    with open(cmd_args.input_json, 'r') as json_file:
        task_list = json.load(json_file)

    model_setup = initialize_qwen_model(cmd_args.model_path)
    inference_results = []
    
    for task_idx, task_item in enumerate(tqdm(task_list, desc='执行推理')):
        img_file_path = os.path.join(cmd_args.image_dir, task_item['image'])
        user_query = task_item['text']
        task_identifier = task_item.get('id', f'task_{task_idx+1}')
        
        try:
            model_response = execute_inference(user_query, img_file_path, model_setup)
        except Exception as error:
            model_response = f"[处理错误] {error}"
        
        source_image = Image.open(img_file_path)
        coordinate_info, point_array = extract_coordinate_data(model_response, source_image.width, source_image.height)
        
        visualization_file = os.path.join(cmd_args.output_dir, f"{task_identifier}_visualization.png")
        create_visualization(img_file_path, coordinate_info, visualization_file, user_query)
        
        inference_results.append({
            "task_id": task_identifier,
            "image_file": task_item['image'],
            "user_question": user_query,
            "model_output": model_response,
            "coordinates": coordinate_info,
            "visualization_path": visualization_file
        })
    
    results_file = os.path.join(cmd_args.output_dir, 'inference_results.json')
    with open(results_file, 'w') as output_file:
        json.dump(inference_results, output_file, indent=2, ensure_ascii=False)
    
    print(f"批量推理完成，结果已保存至 {cmd_args.output_dir}")


if __name__ == "__main__":
    run_batch_inference()