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
from torch.nn.utils.rnn import pad_sequence

from transformers import TrainingArguments, Trainer

from util.dataset import load_classes
from util.io import load_json, store_json, load_text
from dataset.datasets import get_datasets
import pickle