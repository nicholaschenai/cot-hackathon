import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the evaluation script with specified options.")
    parser.add_argument('--option', type=str, default='remote', choices=['remote', 'local'], help='Option to use remote or local model')
    parser.add_argument('--strategy', type=str, default='react', choices=['react', 'zero_shot'], help='Strategy to use for the model')
    parser.add_argument('--num_samples', type=int, default=101, help='Number of samples to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for processing samples')  # Added argument
    return parser.parse_args()