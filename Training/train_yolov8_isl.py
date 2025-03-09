

from ultralytics import YOLO
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def create_training_plots(results):
    # Create a figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training metrics
    epochs = range(1, len(results.results_dict['metrics/precision(B)']) + 1)
    
    ax1.plot(epochs, results.results_dict['metrics/precision(B)'], 'b-', label='Precision')
    ax1.plot(epochs, results.results_dict['metrics/recall(B)'], 'r-', label='Recall')
    ax1.set_title('Precision & Recall')
    ax1.set_xlabel('Epochs')
    ax1.legend()
    
    ax2.plot(epochs, results.results_dict['metrics/mAP50(B)'], 'g-')
    ax2.set_title('mAP50')
    ax2.set_xlabel('Epochs')
    
    ax3.plot(epochs, results.results_dict['train/box_loss'], 'y-')
    ax3.set_title('Box Loss')
    ax3.set_xlabel('Epochs')
    
    ax4.plot(epochs, results.results_dict['train/cls_loss'], 'm-')
    ax4.set_title('Class Loss')
    ax4.set_xlabel('Epochs')
    
    plt.tight_layout()
    plt.show()

def train_yolo_detector():
    # Initialize model
    model = YOLO('yolov8m.pt')
    
    # Training configuration
    training_params = {
        'task': 'detect',
        'data': './datam/data.yaml',
        'epochs': 25,
        'imgsz': 800,
        'plots': True,
        'save': True,
        'cache': True,
        'device': 0,  # GPU device (use -1 for CPU)
        'workers': 8,
        'project': 'yolo_training',
        'name': f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'auto',
        'verbose': True,
        'seed': 42,
        'deterministic': True
    }
    
    # Start training
    print("\n" + "="*50)
    print("Starting YOLOv8 Training")
    print("="*50 + "\n")
    
    results = model.train(**training_params)
    
    print("\n" + "="*50)
    print("Training Completed!")
    print("="*50 + "\n")
    
    # Generate and display training plots
    create_training_plots(results)
    
    return results, model

if __name__ == "__main__":
    results, trained_model = train_yolo_detector()
    
    # Save the model
    save_path = os.path.join('trained_models', 'best_model.pt')
    os.makedirs('trained_models', exist_ok=True)
    trained_model.export(format='pt', path=save_path)