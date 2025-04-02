import os
import hydra
import omegaconf
import torch
import rich.syntax
import rich.tree
import lightning as L

import dataloader
import diffusion
import utils

# Register OmegaConf resolvers
omegaconf.OmegaConf.register_new_resolver('cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver('device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver('eval', eval)
omegaconf.OmegaConf.register_new_resolver('div_up', lambda x, y: (x + y - 1) // y)

def load_model_from_checkpoint(config, tokenizer):
    """Load model from checkpoint."""
    if 'hf' in config.backbone:
        return diffusion.Diffusion(
            config, tokenizer=tokenizer).to('cuda')
    
    # Check if checkpoint path is absolute
    checkpoint_path = config.eval.checkpoint_path
    if not os.path.isabs(checkpoint_path) and not os.path.exists(checkpoint_path):
        # Try looking in the current directory
        if os.path.exists(os.path.join(os.getcwd(), checkpoint_path)):
            checkpoint_path = os.path.join(os.getcwd(), checkpoint_path)
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    
    return diffusion.Diffusion.load_from_checkpoint(
        checkpoint_path,
        tokenizer=tokenizer,
        config=config)

@L.pytorch.utilities.rank_zero_only
def print_config(config: omegaconf.DictConfig, resolve: bool = True) -> None:
    """Prints content of DictConfig using Rich library and its tree structure."""
    style = 'dim'
    tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(
                config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
    rich.print(tree)

def generate_text(config, tokenizer):
    """Generate text samples using the diffusion model."""
    print("Loading model from checkpoint...")
    model = load_model_from_checkpoint(config=config, tokenizer=tokenizer)
    
    if hasattr(model, 'ema') and model.ema is not None:
        print("Using EMA weights for generation")
    
    print(f"Generating {config.sampling.num_sample_batches} batches with batch size {config.loader.eval_batch_size}")
    print(f"Using {config.sampling.steps} diffusion steps with {config.sampling.predictor} predictor")
    
    if config.sampling.semi_ar:
        stride_length = config.sampling.stride_length
        num_strides = config.sampling.num_strides
        print(f"Using semi-autoregressive sampling with stride_length={stride_length}, num_strides={num_strides}")
        
        _, intermediate_samples, _ = model.restore_model_and_semi_ar_sample(
            stride_length=stride_length,
            num_strides=num_strides,
            dt=1 / config.sampling.steps)
        text_samples = intermediate_samples[-1]
    else:
        print("Using standard diffusion sampling")
        samples = model.restore_model_and_sample(
            num_steps=config.sampling.steps)
        text_samples = model.tokenizer.batch_decode(samples)
    
    # Print generated samples
    print("\n" + "="*50)
    print("GENERATED TEXT SAMPLES:")
    print("="*50)
    for i, sample in enumerate(text_samples):
        print(f"\nSample {i+1}:")
        print("-"*50)
        print(sample)
        print("-"*50)
    
    return text_samples

@hydra.main(version_base=None, config_path='configs', config_name='config')
def main(config):
    """Main entry point for text generation."""
    # Override config with command line parameters for unconditional generation
    config.mode = "sample_eval"
    config.model.length = 1024
    config.sampling.predictor = "ddpm_cache"
    config.sampling.steps = 10000
    config.loader.eval_batch_size = 1
    config.sampling.num_sample_batches = 1
    config.backbone = "dit"
    
    # Important: disable hydra's output directory creation to avoid path issues
    os.environ["HYDRA_FULL_ERROR"] = "1"
    
    # Set random seed for reproducibility
    L.seed_everything(config.seed)
    
    # Print the configuration
    print_config(config, resolve=True)
    
    # Get logger and tokenizer
    logger = utils.get_logger(__name__)
    logger.info("Loading tokenizer and generating text samples")
    
    # Get tokenizer without loading dataset
    tokenizer = dataloader.get_tokenizer(config)
    
    # Generate text samples
    text_samples = generate_text(config, tokenizer)
    
    return text_samples

import sys 

sys.argv = [sys.argv[0], 'eval.checkpoint_path=/users/eleves-a/2022/amine.chraibi/mdlm/mdlm.ckpt']
if __name__ == "__main__":
    # Use current working directory for hydra to avoid path issues with outputs
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main() 