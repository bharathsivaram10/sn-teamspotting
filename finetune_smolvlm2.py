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
from torch.nn.utils.rnn import pad_sequence

from transformers import TrainingArguments, Trainer

from util.dataset import load_classes
from util.io import load_json, store_json, load_text
from dataset.datasets import get_datasets
import pickle

actions = {"PASS", "DRIVE", "HEADER", "HIGH PASS", "OUT", "CROSS", "THROW IN", "SHOT", "BALL PLAYER BLOCK", "PLAYER SUCCESSFUL TACKLE", "FREE KICK", "GOAL"}

# Constants
STRIDE = 1
STRIDE_SN = 12
STRIDE_SNB = 2
OVERLAP = 0.9
OVERLAP_SN = 0.50

def convert_pytorch_to_hf_dataset(framepath2label, label2teamaction):
    data_dict = {"frame_path": [], "teamaction": []}
    print("Converting PyTorch dataset to HuggingFace dataset...")
    
    for frame_path, label in framepath2label.items():
        data_dict['frame_path'].append(frame_path)
        data_dict['teamaction'].append(label2teamaction[label])
        
    # Define features correctly - these are individual values, not sequences
    features = Features({
        "frame_path": Value("string"),  # Single frame path as string
        "teamaction": Value("string"),  # Single label as string
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
    
    system_prompt = '''You are a soccer video assistant, and your job is to identify key soccer actions. Here are the list of possible actions: PASS, DRIVE
    HEADER, HIGH PASS, OUT, CROSS, THROW IN, SHOT, BALL PLAYER BLOCK, PLAYER SUCCESSFUL TACKLE, FREE KICK, GOAL, NONE'''
    
    user_prompt = "Identify whether there was an action taken, and if so, what team (left or right). Return in the format 'ACTION-team'. If no action is taken return 'NONE'"
   
    def collate_fn(examples):
        texts = []
        images = []
        
        for example in examples:
            # Get and prepare the image
            image_path = example["frame_path"]
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found: {image_path}")
            
            image = load_image(image_path)
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            label = example["teamaction"]
            
            # Create messages similar to your example format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": system_prompt},
                        {"type": "image"},
                        {"type": "text", "text": user_prompt}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": label}
                    ]
                }
            ]
            
            # Process with apply_chat_template
            text = processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            
            # Important: Wrap the image in a list as shown in your example
            images.append([image])
        
        # Process the batch with nested images list
        batch = processor(
            text=texts,
            images=images,  # This format matches your working example
            return_tensors="pt",
            padding=True
        )
        
        # Handle labels
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        batch["labels"] = labels
        
        return batch
        
    return collate_fn

def get_trainer(args, model, train_ds, val_ds, collate_fn):

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
        do_eval=args.do_eval,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        output_dir=f"./{model_name}-sn-teamspotting",
        hub_model_id=f"{model_name}-sn-teamspotting",
        remove_unused_columns=False,
        report_to=args.report_to,
        dataloader_pin_memory=False,
        gradient_checkpointing=args.gradient_checkpointing,
        load_best_model_at_end = args.load_best_model_at_end,
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_ds,
    eval_dataset=val_ds
    )

    return trainer

def get_model_processor(args):

    model_id = args.model_id

    processor = AutoProcessor.from_pretrained(model_id)

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
            torch_dtype=torch.bfloat16,
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
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, required=False)
    parser.add_argument('--gradient_checkpointing', type=bool, default=True, required=False)
    parser.add_argument('--warmup_steps', type=int, default=50, required=False)
    parser.add_argument('--learning_rate', type=float, default=1e-4, required=False)
    parser.add_argument('--weight_decay', type=float, default=0.01, required=False)
    parser.add_argument('--logging_steps', type=int, default=25, required=False)
    parser.add_argument('--save_strategy', type=str, default="steps", required=False)
    parser.add_argument('--save_steps', type=int, default=5000, required=False)
    parser.add_argument('--save_total_limit', type=int, default=1, required=False)
    parser.add_argument('--optim', type=str, default="adamw_torch", required=False)
    parser.add_argument('--bf16', type=bool, default=True, required=False)
    parser.add_argument('--fp16', type=bool, default=False, required=False)
    parser.add_argument('--report_to', type=str, default="wandb", required=False)
    parser.add_argument('--do_eval', type=bool, default=True, required=False)
    parser.add_argument('--eval_strategy', type=str, default="steps", required=False)
    parser.add_argument('--eval_steps', type=int, default=5000, required=False)
    parser.add_argument('--load_from_pkl', type=bool, default=True, required=False)
    parser.add_argument('--load_best_model_at_end', type=bool, default=True, required=False)
    parser.add_argument('--sample_data', type=bool, default=False, required=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--model', type=str, required=True)
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

    if args.crop_dim <= 0:
        args.crop_dim = None

    classes, joint_train_classes, train_data, val_data, val_data_frames = get_datasets(args)

    if args.store_mode == 'store':
        print('Datasets have been stored correctly! Stop training here and rerun.')
        sys.exit('Datasets have correctly been stored! Stop training here and rerun with load mode.')
    else:
        print('Datasets have been loaded from previous versions correctly!')

    teamaction2label = load_classes("data/soccernetball/class.txt", event_team=True)
    label2teamaction = {v:k for k,v in teamaction2label.items()}
    teamaction2label['NONE'] = 0
    label2teamaction[0] = 'NONE'

    framepath2label = {}

    if args.load_from_pkl:
        for dataset in ['train','val']:
            if args.load_from_pkl:
                with open(f'/home/ubuntu/save_dir/StoreClips/soccernetball/framepath2label_{dataset}.pkl', 'rb') as f:
                    framepath2label[dataset] = pickle.load(f)
    else:
        framepath2label['train'] = train_data.get_paths_labels_dict('train')
        framepath2label['val'] = val_data.get_paths_labels_dict('val')   

    train_ds_hf = convert_pytorch_to_hf_dataset(framepath2label['train'], label2teamaction)
    val_ds_hf = convert_pytorch_to_hf_dataset(framepath2label['val'], label2teamaction)


    if args.sample_data:
        # Randomly sample 10 images and corresponding labels from hugging face dataset and save to folder to check:
        # Sample 10 random indices
        output_dir = "/home/ubuntu/sampled_images"
        os.makedirs(output_dir, exist_ok=True)

        label_file_path = os.path.join(output_dir, "labels.txt")

        sample_indices = random.sample(range(len(train_ds_hf)), 10)

        with open(label_file_path, "w") as f:
            for i, idx in enumerate(sample_indices):
                sample = train_ds_hf[idx]
                image_path = sample["frame_path"]
                label = sample["teamaction"]

                # Save image
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image not found: {image_path}")
                    
                image = load_image(image_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                save_path = os.path.join(output_dir, f"sample_{i}.jpg")
                image.save(save_path)

                f.write(f"sample_{i}.jpg: {label}\n")

        sys.exit(f"Saved 10 images and labels to '{output_dir}'. Check the data and rerun with sample_data option off")

    model, processor = get_model_processor(args)

    collate_fn = get_collate_fn(processor)

    trainer = get_trainer(args, model, train_ds_hf, val_ds_hf, collate_fn)

    trainer.train()

if __name__ == '__main__':
    main(get_args())