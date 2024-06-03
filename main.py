import torch
import argparse
from model import DANet
from dataset import get_dataset
from training import training
from evaluate import evaluate
from utils import same_seeds

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a DANet model.")
    
    # Hyperparameters settings
    parser.add_argument('--size', type=int, default=224, help='Input image size')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--seed', type=int, default=60, help='Random seed')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['sgd', 'adam', 'lbfgs'], help='Optimizer to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--in_channels', type=int, default=512, help='Number of input channels for the model')
    parser.add_argument('--out_channels', type=int, default=128, help='Number of output channels for the model')
    parser.add_argument('--out_dim', type=int, default=8, help='Output dimension for the model')
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    same_seeds(args.seed)
    
    # Get dataset
    train_loader, test_loader = get_dataset(args.size, args.batch_size, args.seed)
    
    # Initialize the model
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"You're now using {device} device")
    model = DANet(args.in_channels, args.out_channels, args.out_dim)
    model.to(device)
    
    # Train the model
    model, history = training(train_loader, model, args.epochs, args.learning_rate, device, args.optimizer)
    
    # Evaluate the model
    evaluate(test_loader, model, history, device, "DANet Model")

if __name__ == "__main__":
    main()