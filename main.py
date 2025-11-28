#!/usr/bin/env python3
"""
Fine-tuning script for Gemma-270M on custom datasets
Supports multi-GPU and multi-node training
"""

from dotenv import load_dotenv
import os

# Load .env file at the start of your script
load_dotenv()

# Now you can access the token
os.environ['HF_HOME']= os.getenv("HF_DIRECTORY")
os.environ['HF_TOKEN']= os.getenv("HF_ACCESS_TOKEN")

# Import standard library modules
import argparse  # For parsing command line arguments
import logging  # For logging training progress and debugging
from typing import Optional  # For type hints with optional values

# Import Pydantic for data validation and settings management
from pydantic import BaseModel, Field  # For creating validated configuration classes

# Import PyTorch for deep learning operations
import torch

# Import HuggingFace datasets library for data loading and processing
from datasets import load_dataset  # For loading datasets from HuggingFace hub or local files

# Import HuggingFace transformers components for model training
from transformers import (
    AutoTokenizer,  # Automatically loads the correct tokenizer for a model
    AutoModelForCausalLM,  # Automatically loads causal language models like GPT/Gemma
    TrainingArguments,  # Configuration class for training hyperparameters
    Trainer,  # High-level training loop handler
    DataCollatorForLanguageModeling,  # Handles batching and padding for language modeling
    set_seed,  # Sets random seeds for reproducibility
)
from transformers.trainer_utils import get_last_checkpoint  # Helper to find most recent checkpoint

# Setup logging configuration for the entire script
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",  # Log format: timestamp - level - logger name - message
    datefmt="%m/%d/%Y %H:%M:%S",  # Date format: month/day/year hour:minute:second
    level=logging.INFO,  # Set logging level to INFO (shows info, warning, and error messages)
)
logger = logging.getLogger(__name__)  # Create a logger instance for this module


class ModelArguments(BaseModel):
    """Arguments for model configuration with Pydantic validation"""
    model_name_or_path: str = Field(  # Path or HuggingFace model ID
        default="google/gemma-270m",  # Default to Google's Gemma 270M parameter model
        description="Path to pretrained model or model identifier from huggingface.co/models"  # Help text for CLI
    )
    use_flash_attention: bool = Field(  # Whether to use Flash Attention 2 optimization
        default=False,  # Disabled by default (requires additional package)
        description="Whether to use flash attention 2"  # Help text explaining this enables FA2
    )

    class Config:
        """Pydantic model configuration"""
        arbitrary_types_allowed = True  # Allow arbitrary types like torch tensors


class DataArguments(BaseModel):
    """Arguments for data configuration with Pydantic validation"""
    dataset_name: Optional[str] = Field(  # Name of dataset from HuggingFace hub
        default=None,  # No default, user must specify or use local files
        description="The name of the dataset to use (via the datasets library)"  # Help text
    )
    dataset_config_name: Optional[str] = Field(  # Configuration/subset name for the dataset
        default=None,  # No default configuration
        description="The configuration name of the dataset to use"  # Help text
    )
    train_file: Optional[str] = Field(  # Path to local training data file
        default=None,  # No default, use HuggingFace dataset if not provided
        description="Path to local training data file (json/jsonl/csv/txt)"  # Supported formats
    )
    validation_file: Optional[str] = Field(  # Path to local validation data file
        default=None,  # No default, training only if not provided
        description="Path to local validation data file"  # Help text
    )
    max_seq_length: int = Field(  # Maximum sequence length in tokens
        default=2048,  # Default to 2048 tokens (common for modern LLMs)
        gt=0,  # Must be greater than 0 (Pydantic validation)
        description="Maximum sequence length for training"  # Help text
    )
    preprocessing_num_workers: int = Field(  # Number of parallel processes for data preprocessing
        default=4,  # Use 4 CPU cores by default
        ge=1,  # Must be greater than or equal to 1 (Pydantic validation)
        description="Number of processes for preprocessing"  # Help text
    )
    text_column: str = Field(  # Name of the column containing text data
        default="text",  # Standard column name used in most datasets
        description="Column name containing the text data"  # Help text
    )

    class Config:
        """Pydantic model configuration"""
        arbitrary_types_allowed = True  # Allow arbitrary types like torch tensors


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Finetune Gemma-270M on custom dataset")  # Create argument parser with description

    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="google/gemma-270m",  # Model identifier argument
                        help="Model identifier from HuggingFace")  # Help text
    parser.add_argument("--use_flash_attention", action="store_true",  # Boolean flag (presence = True)
                        help="Use Flash Attention 2 for faster training")  # Help text

    # Data arguments
    parser.add_argument("--dataset_name", type=str, default=None,  # HuggingFace dataset name
                        help="HuggingFace dataset name")  # Help text
    parser.add_argument("--dataset_config_name", type=str, default=None,  # Dataset configuration/subset
                        help="Dataset configuration name")  # Help text
    parser.add_argument("--train_file", type=str, default=None,  # Path to local training file
                        help="Path to training file")  # Help text
    parser.add_argument("--validation_file", type=str, default=None,  # Path to local validation file
                        help="Path to validation file")  # Help text
    parser.add_argument("--max_seq_length", type=int, default=2048,  # Maximum sequence length
                        help="Maximum sequence length")  # Help text
    parser.add_argument("--text_column", type=str, default="text",  # Column name with text data
                        help="Column name with text data")  # Help text

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./outputs",  # Directory to save checkpoints and outputs
                        help="Output directory for checkpoints")  # Help text
    parser.add_argument("--num_train_epochs", type=int, default=3,  # Number of complete passes through training data
                        help="Number of training epochs")  # Help text
    parser.add_argument("--per_device_train_batch_size", type=int, default=4,  # Batch size for each GPU during training
                        help="Batch size per GPU for training")  # Help text
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,  # Batch size for each GPU during evaluation
                        help="Batch size per GPU for evaluation")  # Help text
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,  # Number of steps to accumulate gradients before updating
                        help="Number of gradient accumulation steps")  # Help text
    parser.add_argument("--learning_rate", type=float, default=2e-5,  # Step size for optimizer (0.00002)
                        help="Learning rate")  # Help text
    parser.add_argument("--weight_decay", type=float, default=0.01,  # L2 regularization strength
                        help="Weight decay")  # Help text
    parser.add_argument("--warmup_steps", type=int, default=500,  # Number of steps to gradually increase learning rate
                        help="Number of warmup steps")  # Help text
    parser.add_argument("--logging_steps", type=int, default=10,  # Log metrics every N steps
                        help="Logging frequency")  # Help text
    parser.add_argument("--save_steps", type=int, default=500,  # Save checkpoint every N steps
                        help="Checkpoint save frequency")  # Help text
    parser.add_argument("--eval_steps", type=int, default=500,  # Evaluate every N steps
                        help="Evaluation frequency")  # Help text
    parser.add_argument("--save_total_limit", type=int, default=3,  # Maximum number of checkpoints to keep (deletes oldest)
                        help="Maximum number of checkpoints to keep")  # Help text
    parser.add_argument("--seed", type=int, default=42,  # Random seed for reproducibility
                        help="Random seed")  # Help text
    parser.add_argument("--bf16", action="store_true",  # Use bfloat16 mixed precision (flag)
                        help="Use bfloat16 precision")  # Help text
    parser.add_argument("--fp16", action="store_true",  # Use float16 mixed precision (flag)
                        help="Use float16 precision")  # Help text
    parser.add_argument("--gradient_checkpointing", action="store_true",  # Trade compute for memory (flag)
                        help="Use gradient checkpointing to save memory")  # Help text
    parser.add_argument("--report_to", type=str, default="tensorboard",  # Where to log metrics (tensorboard/wandb/none)
                        help="Reporting tool (tensorboard/wandb/none)")  # Help text
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,  # Path to checkpoint to resume from
                        help="Path to checkpoint to resume from")  # Help text

    return parser.parse_args()  # Parse the command line arguments and return namespace object


def load_and_prepare_dataset(args, tokenizer):
    """Load and prepare the dataset"""
    logger.info("Loading dataset...")  # Log that we're starting to load data

    # Load dataset from HuggingFace or local files
    if args.dataset_name is not None:  # Check if user specified a HuggingFace dataset name
        # Load from HuggingFace hub
        raw_datasets = load_dataset(  # Download and load dataset from HuggingFace
            args.dataset_name,  # Dataset name (e.g., "wikitext")
            args.dataset_config_name,  # Configuration name (e.g., "wikitext-2-raw-v1")
        )
    else:  # User did not specify dataset name, so load from local files
        # Load from local files
        data_files = {}  # Create empty dictionary to store file paths
        if args.train_file is not None:  # Check if training file path was provided
            data_files["train"] = args.train_file  # Add training file to dictionary
        if args.validation_file is not None:  # Check if validation file path was provided
            data_files["validation"] = args.validation_file  # Add validation file to dictionary

        # Determine file extension to know how to parse the file
        extension = args.train_file.split(".")[-1] if args.train_file else "txt"  # Get file extension (last part after .)
        if extension == "txt":  # Check if extension is txt
            extension = "text"  # HuggingFace datasets library uses "text" instead of "txt"

        raw_datasets = load_dataset(extension, data_files=data_files)  # Load dataset using detected format

    logger.info(f"Dataset loaded: {raw_datasets}")  # Log the dataset structure (splits and sizes)

    # Preprocessing function to tokenize text
    def tokenize_function(examples):
        """Tokenize the text in each example"""
        # Tokenize the texts
        tokenized = tokenizer(  # Apply tokenizer to convert text to token IDs
            examples[args.text_column],  # Get text from specified column
            truncation=True,  # Truncate sequences longer than max_length
            max_length=args.max_seq_length,  # Maximum length to truncate to
            padding=False,  # Don't pad sequences (will be done by data collator later)
            return_special_tokens_mask=False,  # Don't return mask for special tokens
        )
        return tokenized  # Return dictionary with input_ids, attention_mask, etc.

    # Tokenize all datasets
    logger.info("Tokenizing dataset...")  # Log that tokenization is starting
    tokenized_datasets = raw_datasets.map(  # Apply tokenize_function to all examples
        tokenize_function,  # Function to apply to each batch
        batched=True,  # Process in batches for efficiency
        num_proc=args.preprocessing_num_workers if hasattr(args, 'preprocessing_num_workers') else 4,  # Number of parallel processes
        remove_columns=raw_datasets["train"].column_names,  # Remove original columns (keep only tokenized outputs)
        desc="Tokenizing dataset",  # Description to show in progress bar
    )

    # Group texts into chunks of max_seq_length
    def group_texts(examples):
        """Concatenate all texts and split into chunks of max_seq_length"""
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}  # Flatten all sequences into one long sequence per key
        total_length = len(concatenated_examples[list(examples.keys())[0]])  # Get total length of concatenated sequence

        # Drop the small remainder that doesn't fill a complete chunk
        if total_length >= args.max_seq_length:  # Only proceed if we have at least one full chunk
            total_length = (total_length // args.max_seq_length) * args.max_seq_length  # Round down to multiple of max_seq_length

        # Split by chunks of max_seq_length
        result = {  # Create dictionary with chunked sequences
            k: [t[i: i + args.max_seq_length] for i in range(0, total_length, args.max_seq_length)]  # Split into chunks
            for k, t in concatenated_examples.items()  # Apply to all keys (input_ids, attention_mask, etc.)
        }
        return result  # Return dictionary with list of fixed-length chunks

    logger.info("Grouping texts...")  # Log that grouping is starting
    lm_datasets = tokenized_datasets.map(  # Apply group_texts function to create fixed-length chunks
        group_texts,  # Function to apply
        batched=True,  # Process in batches
        num_proc=args.preprocessing_num_workers if hasattr(args, 'preprocessing_num_workers') else 4,  # Number of parallel processes
        desc="Grouping texts",  # Description for progress bar
    )

    train_dataset = lm_datasets["train"]  # Extract training dataset split
    eval_dataset = lm_datasets.get("validation", None)  # Extract validation split if it exists, otherwise None

    logger.info(f"Training samples: {len(train_dataset)}")  # Log number of training examples
    if eval_dataset:  # Check if validation dataset exists
        logger.info(f"Validation samples: {len(eval_dataset)}")  # Log number of validation examples

    return train_dataset, eval_dataset  # Return both datasets (eval_dataset may be None)


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()  # Parse command line arguments into args object

    # Set seed for reproducibility
    set_seed(args.seed)  # Set random seeds for Python, NumPy, and PyTorch

    # Setup logging
    logger.info(f"Training arguments: {args}")  # Log all arguments for debugging

    # Detect last checkpoint in output directory
    last_checkpoint = None  # Initialize to None
    if os.path.isdir(args.output_dir):  # Check if output directory exists
        last_checkpoint = get_last_checkpoint(args.output_dir)  # Find most recent checkpoint folder
        if last_checkpoint is not None and args.resume_from_checkpoint is None:  # If checkpoint found and no explicit resume path
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}")  # Log that we'll resume from this checkpoint

    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.model_name_or_path}")  # Log tokenizer loading
    tokenizer = AutoTokenizer.from_pretrained(  # Load tokenizer that matches the model
        args.model_name_or_path,  # Model identifier (same as model)
        trust_remote_code=True, # Allow execution of custom code in tokenizer (needed for some models)
    )

    # Set padding token if not set (required for batching)
    if tokenizer.pad_token is None:  # Check if tokenizer has no padding token defined
        tokenizer.pad_token = tokenizer.eos_token  # Use end-of-sequence token as padding token

    # Load model
    logger.info(f"Loading model from {args.model_name_or_path}")  # Log model loading
    model_kwargs = {  # Create dictionary of keyword arguments for model loading
        "trust_remote_code": True,  # Allow execution of custom code in model (needed for some models)
    }

    if args.use_flash_attention:  # Check if user enabled Flash Attention 2
        model_kwargs["attn_implementation"] = "flash_attention_2"  # Use FA2 for attention computation
        model_kwargs["torch_dtype"] = torch.bfloat16  # FA2 requires bfloat16 dtype

    if args.gradient_checkpointing:  # Check if user enabled gradient checkpointing
        model_kwargs["use_cache"] = False  # Disable KV cache (incompatible with gradient checkpointing)

    model = AutoModelForCausalLM.from_pretrained(  # Load pretrained causal language model
        args.model_name_or_path,  # Model identifier
        **model_kwargs,  # Unpack model configuration kwargs
    )

    if args.gradient_checkpointing:  # Check if gradient checkpointing is enabled
        model.gradient_checkpointing_enable()  # Enable gradient checkpointing to save memory

    # Resize token embeddings if needed (if tokenizer vocab is larger than model vocab)
    embedding_size = model.get_input_embeddings().weight.shape[0]  # Get current size of embedding matrix
    if len(tokenizer) > embedding_size:  # Check if tokenizer has more tokens than model
        model.resize_token_embeddings(len(tokenizer))  # Resize embedding matrix to match tokenizer

    # Load and prepare dataset
    train_dataset, eval_dataset = load_and_prepare_dataset(args, tokenizer)  # Load, tokenize, and chunk the data

    # Data collator for dynamic padding and label creation
    data_collator = DataCollatorForLanguageModeling(  # Create data collator for batching
        tokenizer=tokenizer,  # Tokenizer for padding
        mlm=False,  # Causal LM, not masked LM (don't randomly mask tokens)
    )

    # Training arguments configuration
    training_args = TrainingArguments(  # Create training configuration object
        output_dir=args.output_dir,  # Where to save checkpoints and logs
        num_train_epochs=args.num_train_epochs,  # Number of complete passes through training data
        per_device_train_batch_size=args.per_device_train_batch_size,  # Batch size per GPU for training
        per_device_eval_batch_size=args.per_device_eval_batch_size,  # Batch size per GPU for evaluation
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # Number of steps to accumulate before updating
        learning_rate=args.learning_rate,  # Optimizer learning rate
        weight_decay=args.weight_decay,  # L2 regularization coefficient
        warmup_steps=args.warmup_steps,  # Number of warmup steps for learning rate scheduler
        logging_steps=args.logging_steps,  # Log metrics every N steps
        save_steps=args.save_steps,  # Save checkpoint every N steps
        eval_steps=args.eval_steps if eval_dataset else None,  # Evaluate every N steps (only if validation data exists)
        save_total_limit=args.save_total_limit,  # Maximum number of checkpoints to keep
        bf16=args.bf16,  # Use bfloat16 mixed precision
        fp16=args.fp16,  # Use float16 mixed precision
        gradient_checkpointing=args.gradient_checkpointing,  # Enable gradient checkpointing
        # report_to=args.report_to,  # Where to report metrics (tensorboard/wandb/none)
        do_eval=eval_dataset is not None,  # Enable evaluation if validation data exists
        eval_strategy="steps" if eval_dataset else "no",  # Evaluate every N steps or never
        load_best_model_at_end=True if eval_dataset else False,  # Load best checkpoint at end (only if evaluating)
        metric_for_best_model="loss" if eval_dataset else None,  # Use loss to determine best model
        greater_is_better=False if eval_dataset else None,  # Lower loss is better
        ddp_find_unused_parameters=False,  # Don't search for unused parameters in DDP (faster)
        seed=args.seed,  # Random seed for training
    )

    # Initialize Trainer
    trainer = Trainer(  # Create Trainer object to handle training loop
        model=model,  # Model to train
        args=training_args,  # Training configuration
        train_dataset=train_dataset,  # Training data
        eval_dataset=eval_dataset,  # Validation data (can be None)
        tokenizer=tokenizer,  # Tokenizer (for saving with model)
        data_collator=data_collator,  # Function to batch and pad data
    )

    # Training
    logger.info("Starting training...")  # Log that training is beginning
    checkpoint = None  # Initialize checkpoint path to None
    if args.resume_from_checkpoint is not None:  # Check if user specified explicit checkpoint to resume from
        checkpoint = args.resume_from_checkpoint  # Use user-specified checkpoint
    elif last_checkpoint is not None:  # Check if we found a checkpoint in output directory
        checkpoint = last_checkpoint  # Use auto-detected checkpoint

    train_result = trainer.train(resume_from_checkpoint=checkpoint)  # Run training loop, optionally resuming from checkpoint

    # Save model
    logger.info("Saving final model...")  # Log that we're saving the model
    trainer.save_model()  # Save model weights and configuration to output_dir

    # Save training metrics
    metrics = train_result.metrics  # Get training metrics (loss, steps, etc.)
    trainer.log_metrics("train", metrics)  # Log metrics to console
    trainer.save_metrics("train", metrics)  # Save metrics to JSON file
    trainer.save_state()  # Save trainer state (optimizer, scheduler, RNG states)

    # Evaluation
    if eval_dataset is not None:  # Check if validation dataset exists
        logger.info("Running final evaluation...")  # Log that we're doing final evaluation
        metrics = trainer.evaluate()  # Run evaluation on validation set
        trainer.log_metrics("eval", metrics)  # Log evaluation metrics to console
        trainer.save_metrics("eval", metrics)  # Save evaluation metrics to JSON file

    logger.info("Training completed!")  # Log that training finished successfully


if __name__ == "__main__":  # Check if script is being run directly (not imported)
    main()  # Call main function to start training
