
import math
import sys,cv2
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset
import matplotlib.pyplot as plt
import numpy as np
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld
class RedirectStdoutToFile:
    def __init__(self, filename):
        self.filename = filename
        self._stdout = sys.stdout

    def __enter__(self):
        self.file = open(self.filename, 'w')
        sys.stdout = self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._stdout
        if self.file:
            self.file.close()
            
coco_name_dict = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush"
}

# Reversing the dictionary (value to key)
name_to_id = {v: k for k, v in coco_name_dict.items()}

# Function to get ID from name
def get_id_from_name(name):
    return name_to_id.get(name)


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


from torchvision.transforms.functional import to_pil_image

def tensor_to_image(tensor):
    """Converts a torch tensor into a PIL image."""
    # Assuming tensor is in the format [C, H, W]
    return to_pil_image(tensor)

def save_image(tensor, filename):
    """Saves a torch tensor as an image file."""
    img = tensor_to_image(tensor)
    img.save(filename)

def show_image(tensor):
    """Displays a torch tensor as an image."""
    img = tensor_to_image(tensor)
    img.show()


def plot_predictions(input_image, text_queries, scores, boxes, labels):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(input_image, extent=(0, 1, 1, 0))
    ax.set_axis_off()

    for score, box, label in zip(scores, boxes, labels):
      if score < 0.1:
        continue

      cx, cy, w, h = box
      ax.plot([cx-w/2, cx+w/2, cx+w/2, cx-w/2, cx-w/2],
              [cy-h/2, cy-h/2, cy+h/2, cy+h/2, cy-h/2], "r")
      ax.text(
          cx - w / 2,
          cy + h / 2 + 0.015,
          f"{text_queries[label]}: {score:1.2f}",
          ha="left",
          va="top",
          color="red",
          bbox={
              "facecolor": "white",
              "edgecolor": "red",
              "boxstyle": "square,pad=.3"
          })
    

    fig.savefig('owlvit.jpg')

import torch
from PIL import Image

def torch_tensor_to_pil(tensor):
    # Remove the batch dimension if present
    if tensor.ndim == 4 and tensor.shape[0] == 1:  # Assuming batch size of 1
        tensor = tensor.squeeze(0)

    # Check if the tensor is normalized and unnormalize it
    # This step depends on how the tensor was normalized initially
    # Here's an example for tensors in the range [0, 1]
    tensor = tensor.mul(255).byte()
    
    # Convert from channel-first to channel-last format
    tensor = tensor.permute(1, 2, 0)

    # Convert to PIL Image
    return Image.fromarray(tensor.cpu().numpy())


@torch.inference_mode()
def evaluate(args, model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        print('output before ', outputs)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time
        print('output after ', outputs)
        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        print ('res ', res)
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        return

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


images_without_ann = [25593, 41488, 42888, 49091, 58636, 64574, 98497, 101022, 121153, 127135, 173183, 176701, 198915, 200152, 226111, 228771, 240767, 260657, 261796, 267946, 268996, 270386, 278006, 308391, 310622, 312549, 320706, 330554, 344611, 370999, 374727, 382734, 402096, 404601, 447789, 458790, 461275, 476491, 477118, 481404, 502910, 514540, 528977, 536343, 542073, 550939, 556498, 560371]

@torch.inference_mode()
def evaluate_owlvit(args, model, data_loader, device):
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model.to(device)
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    itr = 160000
    for images, targets in metric_logger.log_every(data_loader, 1, header):

        
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        label_names = [coco_name_dict[label.item()] for label in targets[0]['labels']]

        if args.query_option != 'without':
            if args.subset == 'weather':
                text_query = [coco_name_dict[label.item()]+' in extreme' + args.subset for label in targets[0]['labels']]
            elif args.subset == 'handmake':
                text_query = [coco_name_dict[label.item()] + ' handmade' for label in targets[0]['labels']]
            elif args.subset == 'painting':
                text_query = [coco_name_dict[label.item()] + ' in painting' for label in targets[0]['labels']]
            elif args.subset == '':
                text_query = [coco_name_dict[label.item()] for label in targets[0]['labels']]
            else:
                text_query = [coco_name_dict[label.item()]+' ' + args.subset for label in targets[0]['labels']]
        else:
            text_query = [coco_name_dict[label.item()] for label in targets[0]['labels']]
        
        


        # Example normalization parameters (these should be the ones you used during normalization)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Your normalized images tensor
        # Assuming `normalized_images` is a PyTorch tensor of shape [N, C, H, W] 
        # where N is the number of images, C is the number of channels (3 for RGB images), H is height, and W is width.

        # Unnormalize
        unnormalized_images = images[0]
        
        for c in range(3):
            unnormalized_images[:, c] *= std[c]
            unnormalized_images[:, c] += mean[c]

        # If you normalized the images to a 0-1 range and need to convert back to 0-255
        unnormalized_images *= 255

        # Convert to an appropriate type for visualization or saving as an image file
        unnormalized_images = unnormalized_images.type(torch.uint8)
        plot_img = unnormalized_images
        #save_image(unnormalized_images, 'unnormalized_image.jpg')
        #show_image(unnormalized_images)
        if len(text_query) == 0:
            print('no query ..jump to next exmaple')
            continue
        print('key query owlvit', text_query)
        
        if targets[0]['image_id'] not in images_without_ann:

            inputs = processor(text=text_query, images=unnormalized_images, return_tensors="pt").to(device)
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
            
            logits = torch.max(outputs["logits"][0], dim=-1)
            scores = torch.sigmoid(logits.values).cpu().detach().numpy()

            # Get prediction labels and boundary boxes
            labels = logits.indices.cpu().detach().numpy()
            boxes = outputs["pred_boxes"][0].cpu().detach().numpy()
            
            pil_image = torch_tensor_to_pil(images[0])
            target_sizes = torch.tensor([pil_image.size[::-1]])
            results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
            i = 0  # Retrieve predictions for the first image for the corresponding text queries
            boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]
            new_labels = []
            new_labels_names = []
            new_boxes = []
            new_scores = []

            for box, score, label in zip(boxes, scores, labels):
                if score < 0.1:
                    continue

                class_label = label_names[label.item()]
                new_labels_names.append(class_label)
                new_label = get_id_from_name(class_label)
                new_labels.append(new_label)
                box = [round(i, 2) for i in box.tolist()]
                new_boxes.append(box)
                new_scores.append(score.item())

            
            
            output_dict = [{'boxes': torch.tensor(new_boxes), 'labels': torch.tensor(new_labels), 'scores': torch.tensor(new_scores)}]
            

            if itr == 16:
                from transformers.image_utils import ImageFeatureExtractionMixin
                mixin = ImageFeatureExtractionMixin()
                print ('saving image')
                # Load example image
                image_size = model.config.vision_config.image_size
                image = mixin.resize(pil_image, image_size)
                
                input_image = np.asarray(image).astype(np.float32)//255.0
                

                plot_predictions(Image.fromarray(np.uint8(pil_image)).convert("RGB"), label_names, scores, boxes.cpu(), labels)
                return

            itr += 1


            
            outputs = output_dict
            
            model_time = time.time() - model_time
        

            
            
            print('image_id ', targets[0]['image_id'])
            print('prediction class names ', new_labels_names)    
            print('ground truth ', targets[0])
            
            res = {target["image_id"]: output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        else:
            print('<<<<<<<<<<<<image without annotation>>>>>>>>>>>> ', targets[0]['image_id'])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    with RedirectStdoutToFile(f'{args.name}_dataset_{args.coco_dc_subset}{args.pd_subset}{args.corruption}{args.severity}_results.txt'):
        coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator


@torch.inference_mode()
def evaluate_yolow(args, model, data_loader, device):
    
    model = YOLOWorld(model_id="yolo_world/l")

    '''model.to(device)'''
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    '''model.eval()'''
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)
    itr = 16000
    for images, targets in metric_logger.log_every(data_loader, 1, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        label_names = [coco_name_dict[label.item()] for label in targets[0]['labels']]

        if args.query_option != 'without':
            if args.subset == 'weather':
                text_query = [coco_name_dict[label.item()]+' in extreme' + args.subset for label in targets[0]['labels']]
            elif args.subset == 'handmake':
                text_query = [coco_name_dict[label.item()] + ' handmade' for label in targets[0]['labels']]
            elif args.subset == 'painting':
                text_query = [coco_name_dict[label.item()] + ' in painting' for label in targets[0]['labels']]
            elif args.subset == '':
                text_query = [coco_name_dict[label.item()] for label in targets[0]['labels']]
            else:
                text_query = [coco_name_dict[label.item()]+' ' + args.subset for label in targets[0]['labels']]
        else:
            text_query = [coco_name_dict[label.item()] for label in targets[0]['labels']]
        
        


        # Example normalization parameters (these should be the ones you used during normalization)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Unnormalize
        unnormalized_images = images[0]
        
        for c in range(3):
            unnormalized_images[:, c] *= std[c]
            unnormalized_images[:, c] += mean[c]

        # If you normalized the images to a 0-1 range and need to convert back to 0-255
        unnormalized_images *= 255

        # Convert to an appropriate type for visualization or saving as an image file
        unnormalized_images = unnormalized_images.type(torch.uint8)

        unnormalized_images = unnormalized_images.permute(1, 2, 0).cpu().numpy()  # Assuming image is in RGB
        unnormalized_images = cv2.cvtColor(unnormalized_images, cv2.COLOR_RGB2BGR)

        #save_image(unnormalized_images, 'unnormalized_image.jpg')
        #show_image(unnormalized_images)
        if len(text_query) == 0:
            print('no query ..jump to next exmaple')
            continue
        print('key query ', text_query)
        
        if targets[0]['image_id'] not in images_without_ann:
            model.set_classes(text_query)
            results = model.infer(unnormalized_images)
            detections = sv.Detections.from_inference(results)
            if itr == 16:
                BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=2)
                LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=2, text_scale=1, text_color=sv.Color.BLACK)
                annotated_image = unnormalized_images.copy()
                annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
                annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections)
                sv.plot_image(annotated_image, (10, 10))
                return
            itr += 1
            pred_class_names = [text_query[item] for item in list(detections.class_id)]
            pred_class_labels = [get_id_from_name(classname) for classname in pred_class_names]                                
            # Extract boxes, confidence (scores), and class IDs from the detections
            xyxy_boxes = detections.xyxy
            confidences = detections.confidence
        

            # Convert lists to PyTorch tensors
            boxes_tensor = torch.tensor(xyxy_boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(pred_class_labels, dtype=torch.int64)
            scores_tensor = torch.tensor(confidences, dtype=torch.float32)

            # Combine into the expected output format
            output_coco_format = [{
                'boxes': boxes_tensor,
                'labels': labels_tensor,
                'scores': scores_tensor
            }]
            print('image_id ', targets[0]['image_id'])
            print('prediction class names ', pred_class_names)    
            print('ground truth ', targets[0])
            outputs = output_coco_format
            model_time = time.time() - model_time
                   
            
            #print('outputs ', outputs)
            res = {target["image_id"]: output for target, output in zip(targets, outputs)}
            evaluator_time = time.time()
            coco_evaluator.update(res)
            evaluator_time = time.time() - evaluator_time
            metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        else:
            print('<<<<<<<<<<<<image without annotation>>>>>>>>>>>> ', targets[0]['image_id'])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    with RedirectStdoutToFile(f'{args.name}_dataset_{args.coco_dc_subset}{args.pd_subset}{args.corruption}{args.severity}_results.txt'):
        coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator