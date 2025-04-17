import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn.init as init
from vision.ssd.config import mobilenetv1_ssd_config as config
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite

def _xavier_init_(m):
    if isinstance(m, torch.nn.Conv2d):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

def parse_args():
    parser = argparse.ArgumentParser(description='Export MobileNetV2 SSD Lite model to ONNX')

    parser.add_argument('--input-size', type=int, required=True, 
                        help='Input size (must be divisible by 32)')
    parser.add_argument('--num-classes', type=int, required=True, 
                        help='Number of classes (including background class)')
    parser.add_argument('--output-dir', type=str, required=True, 
                        help='Directory to save the ONNX model')
    parser.add_argument('--model-weights', type=str, default=None,
                        help='Path to model weights (if not provided, default initialization will be used)')
    return parser.parse_args()

def main():
    args = parse_args()

    if args.input_size % 32 != 0:
        raise ValueError(f"Input size must be divisible by 32, got {args.input_size}")

    num_classes = args.num_classes
    print(f"Creating model with {num_classes} classes (including background class)")

    # modify global config
    original_image_size = config.image_size
    config.image_size = args.input_size

    model = create_mobilenetv2_ssd_lite(
        num_classes=num_classes, 
        is_test=False,
        input_size=args.input_size
    )

    if args.model_weights:
        if not os.path.exists(args.model_weights):
            raise FileNotFoundError(f"Model weights file not found: {args.model_weights}")
        print(f"Loading weights from {args.model_weights}")
        model.load(args.model_weights)
    else:
        print("Use default pretrained weights")
        pretrained_state_dict = torch.load("models/mb2-ssd-lite-mp-0_686.pth", 
                                  map_location=torch.device('cpu'),
                                  weights_only=False)
        filtered_dict = {k: v for k, v in pretrained_state_dict.items() 
                if not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        model.load_state_dict(filtered_dict, strict=False)
        model.classification_headers.apply(_xavier_init_)
        model.regression_headers.apply(_xavier_init_)
    model.eval()

    # ONNX export
    dummy_input = torch.randn(1, 3, args.input_size, args.input_size)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_path = os.path.join(args.output_dir, f"mb2-ssd-lite-{args.input_size}.onnx")
    print(f"Exporting model to {output_path}")
    torch.onnx.export(
        model,
        dummy_input, 
        output_path, 
        export_params=True, 
        opset_version=11,
        input_names=['input'],
        output_names=['scores', 'boxes'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'scores': {0: 'batch_size'},
            'boxes': {0: 'batch_size'}
        }
    )
    print(f"Model successfully exported to {output_path}")

    config.image_size = original_image_size

if __name__ == "__main__":
    main()