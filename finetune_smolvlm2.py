#Standard imports
import argparse
import numpy as np
import random
from torch.utils.data import DataLoader
import wandb
import sys
from datasets import Features, Value, Sequence
from datasets import Dataset as HFDataset
from transformers.image_utils import load_image

import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForImageTextToText
import os

from transformers import TrainingArguments, Trainer

from util.dataset import load_classes
from util.io import load_json, store_json, load_text
from dataset.datasets import get_datasets

actions = {"PASS", "DRIVE", "HEADER", "HIGH PASS", "OUT", "CROSS", "THROW IN", "SHOT", "BALL PLAYER BLOCK", "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL"}

# Constants
STRIDE = 1
STRIDE_SN = 12
STRIDE_SNB = 2
OVERLAP = 0.9
OVERLAP_SN = 0.50

def convert_pytorch_to_hf_dataset(pytorch_dataset, label2teamaction):

    data_dict = {"frame_paths" : [], "labels": []}

    print("Converting PyTorch dataset to HuggingFace dataset...")

    for i in range(min(len(pytorch_dataset), pytorch_dataset._dataset_len)):

        # Get frame paths for this sample
        frame_paths = pytorch_dataset._frame_paths[i]
        data_dict["frame_paths"].append(frame_paths)

        # Get labels for this sample
        labels = pytorch_dataset._labels_store[i]
        processed_labels = np.zeros(pytorch_dataset._clip_len, dtype=np.int64)
        
        for label_info in labels:
            label_idx = label_info['label_idx']
            if 0 <= label_idx < pytorch_dataset._clip_len:
                processed_labels[label_idx] = label_info['label']
        
        data_dict["labels"].append(processed_labels)

    # Define features for validation
    features = Features({
        "frame_paths": Sequence(Value("string")),  # List of frame paths
        "labels": Sequence(Value("string")),       # Labels for each frame
    })

    # Create the HuggingFace dataset
    hf_dataset = HFDataset.from_dict(data_dict, features=features)
    
    return hf_dataset

def update_args(args, config):
    #Update arguments with config file
    args.frame_dir = config['frame_dir']
    args.save_dir = config['save_dir'] + '/' + args.model # + '-' + str(args.seed) -> in case multiple seeds
    args.store_dir = os.path.join(config['save_dir'], 'StoreClips', config['dataset']) #where to store clips information
    args.store_mode = config['store_mode']
    args.batch_size = config['batch_size']
    args.clip_len = config['clip_len']
    args.crop_dim = config['crop_dim']
    args.dataset = config['dataset']
    args.event_team = config['event_team']
    args.radi_displacement = config['radi_displacement']
    args.epoch_num_frames = config['epoch_num_frames']
    args.feature_arch = config['feature_arch']
    args.learning_rate = config['learning_rate']
    args.mixup = config['mixup']
    args.modality = config['modality']
    args.num_classes = config['num_classes']
    args.num_epochs = config['num_epochs']
    args.warm_up_epochs = config['warm_up_epochs']
    args.start_val_epoch = config['start_val_epoch']
    args.temporal_arch = config['temporal_arch']
    args.n_layers = config['n_layers']
    args.sgp_ks = config['sgp_ks']
    args.sgp_r = config['sgp_r']
    args.only_test = config['only_test']
    args.criterion = config['criterion']
    args.num_workers = config['num_workers']
    if 'joint_train' in config:
        args.joint_train = config['joint_train']
        args.joint_train['store_dir'] = os.path.join(args.save_dir, 'StoreClips', args.joint_train['dataset'])
    else:
        args.joint_train = None
    return args

def get_collate_fn(processor):
    
    # Get the image token ID for your processor
    image_token_id = processor.tokenizer.additional_special_tokens_ids[
                processor.tokenizer.additional_special_tokens.index("<image>")]
    
    system_prompt = '''You are a soccer video assistant, and your job is to identify key soccer actions. Here are the definitions for each action we are interested in:
    Pass: A player kicks the ball to a teammate to maintain possession
    Drive: An attacking dribble taken
    Header: Striking the ball using the head, usually to pass, clear, or score
    High Pass: A pass lofted through the air to reach a teammate over distance or defenders
    Out: The ball goes completely over the touchline or goal line, stopping play
    Cross: A pass from the side of the field into the opponent's penalty area
    Throw In: A two-handed overhead throw used to return the ball into play after it goes out on the sideline
    Shot: An attempt to score by kicking or heading the ball toward the goal
    Ball Player Block: A player obstructs the ball or ball carrier to prevent progress
    Player Successful Tackle: A player legally takes the ball away from an opponent
    Free Kick: A kick awarded after a foul, allowing an unopposed restart
    Goal: When the entire ball crosses the goal line between the posts and under the crossbar
    None: None of the above actions'''

    user_prompt = "Identify whether there was an action taken, and if so, what team, return in the format 'ACTION-team'. If no action is taken return 'None'"
    
    def collate_fn(examples):
    
        texts = []
        images = []

        for example in examples:

            image_path = os.path.join("visual-spatial-reasoning/images", example["frame_paths"])

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")

            image = load_image(image_path)

            if image.mode != 'RGB':
                image = image.convert('RGB')

            label = example["labels"]

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"{user_prompt}"}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": label}
                    ]
                }
            ]
            text = processor.apply_chat_template(messages, add_generation_prompt=False)

            texts.append(text.strip())
            images.append([image])

        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch
        
    return collate_fn

def get_trainer(args, model, train_ds, collate_fn):

    model_name = args.model_id.split("/")[-1]

    training_args = TrainingArguments(
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        optim=args.optim, # for 8-bit, keep paged_adamw_8bit, else adamw_hf
        bf16=args.bf16,
        output_dir=f"./{model_name}-sn-teamspotting",
        hub_model_id=f"{model_name}-sn-teamspotting",
        remove_unused_columns=False,
        report_to=args.report_to,
        dataloader_pin_memory=False
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_ds,
    )

    return trainer

def get_model_processor(args):

    model_id = args.model_id

    processor = AutoProcessor.from_pretrained(
    model_id
    )

    if args.USE_QLORA or args.USE_LORA:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
            use_dora=False if args.USE_QLORA else True,
            init_lora_weights="gaussian"
        )
        lora_config.inference_mode = False
        if args.USE_QLORA:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=bnb_config if args.USE_QLORA else None,
            _attn_implementation="flash_attention_2",
            device_map="auto"
        )
        model.add_adapter(lora_config)
        model.enable_adapters()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        print(model.get_nb_trainable_parameters())
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
        ).to("cuda")

        # if you'd like to only fine-tune LLM
        for param in model.model.vision_model.parameters():
            param.requires_grad = False

    peak_mem = torch.cuda.max_memory_allocated()
    print(f"The model as is is holding: {peak_mem / 1024**3:.2f} of GPU RAM")

    return model, processor

def get_args():
    #Basic arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--USE_LORA', type=bool, default=False, required=False)
    parser.add_argument('--USE_QLORA', type=bool, default=False, required=False)
    parser.add_argument('--model_id', type=str, default="HuggingFaceTB/SmolVLM2-2.2B-Instruct", required=False)
    parser.add_argument('--num_train_epochs', type=int, default=1, required=False)
    parser.add_argument('--per_device_train_batch_size', type=int, default=2, required=False)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, required=False)
    parser.add_argument('--warmup_steps', type=int, default=50, required=False)
    parser.add_argument('--learning_rate', type=float, default=1e-4, required=False)
    parser.add_argument('--weight_decay', type=float, default=0.01, required=False)
    parser.add_argument('--logging_steps', type=int, default=25, required=False)
    parser.add_argument('--save_strategy', type=str, default="steps", required=False)
    parser.add_argument('--save_steps', type=int, default=250, required=False)
    parser.add_argument('--save_total_limit', type=int, default=1, required=False)
    parser.add_argument('--optim', type=str, default="adamw_hf", required=False)
    parser.add_argument('--bf16', type=bool, default=True, required=False)
    parser.add_argument('--report_to', type=str, default="tensorboard", required=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('-ag', '--acc_grad_iter', type=int, default=1,
                        help='Use gradient accumulation')
    return parser.parse_args()

def main(args):
    print('Setting seed to: ', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config_path = args.model.split('_')[0] + '/' + args.model + '.json'
    config = load_json(os.path.join('config', config_path))
    args = update_args(args, config)

    #Variables for SN & SNB label paths if datastes
    if (args.dataset == 'soccernet') | (args.dataset == 'soccernetball'):
        global LABELS_SN_PATH
        global LABELS_SNB_PATH
        LABELS_SN_PATH = load_text(os.path.join('data', 'soccernet', 'labels_path.txt'))[0]
        LABELS_SNB_PATH = load_text(os.path.join('data', 'soccernetball', 'labels_path.txt'))[0]

    assert args.batch_size % args.acc_grad_iter == 0
    if args.crop_dim <= 0:
        args.crop_dim = None

    classes, joint_train_classes, train_data, val_data, val_data_frames = get_datasets(args)

    if args.store_mode == 'store':
        print('Datasets have been stored correctly! Stop training here and rerun.')
        sys.exit('Datasets have correctly been stored! Stop training here and rerun with load mode.')
    else:
        print('Datasets have been loaded from previous versions correctly!')

    for i in range(len(train_data)):
        sample = train_data[i]
        for key, value in sample.items():
            print(key, value)

    teamaction2label = load_classes("soccernetball/class.txt", event_team=True)
    label2teamaction = {k:v for k,v in teamaction2label.items()}

    print(label2teamaction)
    print(teamaction2label)

    model, processor = get_model_processor(args)

    # train_ds_hf = get_hf_dataset(train_data)

    # collate_fn = get_collate_fn(processor)

    # trainer = get_trainer(args, model, train_data, collate_fn)

    # trainer.train()

if __name__ == '__main__':
    main(get_args())