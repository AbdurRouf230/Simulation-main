"""
Flower Detection and Pollination Status Models.

Extends IntelleSwarm EdgePerceptionTransformer for agricultural applications.
Provides real-time flower detection, species classification, and pollination assessment.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
from dataclasses import dataclass
from enum import Enum

# Import from IntelleSwarm framework
import sys
sys.path.append('..')
from genai_framework.sdk.edge import EdgePerceptionTransformer


class PollinationStatus(Enum):
    """Pollination status of detected flowers."""
    READY = "ready"              # Flower open and ready for pollination
    POLLINATED = "pollinated"    # Already pollinated (pollen visible)
    PAST_PRIME = "past_prime"    # Flower wilted or past optimal time
    BUD = "bud"                  # Flower bud not yet open
    DAMAGED = "damaged"          # Flower damaged or diseased


@dataclass
class FlowerDetection:
    """Single flower detection result."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    species: str
    pollination_status: PollinationStatus
    confidence: float
    center_3d: Optional[Tuple[float, float, float]] = None  # 3D world coordinates
    pollen_source_quality: float = 0.0  # Quality as pollen source (0-1)
    pollen_need_urgency: float = 0.0    # Urgency for receiving pollen (0-1)


class FlowerDetector(nn.Module):
    """
    Advanced flower detection model extending EdgePerceptionTransformer.

    Detects flowers in agricultural environments with high precision
    for autonomous pollination applications.
    """

    def __init__(self,
                 num_species: int = 50,
                 img_channels: int = 3,
                 img_size: Tuple[int, int] = (640, 480),
                 emb_dim: int = 256,
                 num_heads: int = 8,
                 depth: int = 6):
        super().__init__()

        self.num_species = num_species
        self.img_size = img_size

        # Base perception model from IntelleSwarm
        self.base_model = EdgePerceptionTransformer(
            img_channels=img_channels,
            patch_size=16,
            emb_dim=emb_dim,
            num_heads=num_heads,
            depth=depth
        )

        # Agricultural-specific detection heads
        self.flower_detection_head = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 5)  # [x, y, w, h, objectness]
        )

        self.species_classifier = nn.Sequential(
            nn.Linear(emb_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_species)
        )

        self.pollination_status_head = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, len(PollinationStatus))
        )

        # Depth estimation for 3D positioning
        self.depth_estimator = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Depth in meters
        )

        # Agricultural quality assessments
        self.pollen_quality_estimator = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Pollen quality score 0-1
            nn.Sigmoid()
        )

        # Species names mapping
        self.species_names = self._load_species_names()

    def _load_species_names(self) -> List[str]:
        """Load crop species names."""
        return [
            # Tree Fruits
            "apple_malus_domestica", "cherry_prunus_avium", "almond_prunus_dulcis",
            "peach_prunus_persica", "pear_pyrus_communis", "plum_prunus_domestica",
            "apricot_prunus_armeniaca", "citrus_orange", "citrus_lemon", "citrus_lime",

            # Berry Crops
            "blueberry_vaccinium", "strawberry_fragaria", "raspberry_rubus_idaeus",
            "blackberry_rubus", "cranberry_vaccinium_macrocarpon", "elderberry_sambucus",

            # Vegetable Crops
            "tomato_solanum_lycopersicum", "pepper_capsicum_annuum", "eggplant_solanum_melongena",
            "cucumber_cucumis_sativus", "squash_cucurbita_pepo", "pumpkin_cucurbita_maxima",
            "melon_cucumis_melo", "watermelon_citrullus_lanatus", "zucchini_cucurbita_pepo",

            # Legumes
            "bean_phaseolus_vulgaris", "pea_pisum_sativum", "soybean_glycine_max",
            "chickpea_cicer_arietinum", "lentil_lens_culinaris",

            # Herbs and Spices
            "lavender_lavandula", "rosemary_rosmarinus", "thyme_thymus_vulgaris",
            "oregano_origanum_vulgare", "basil_ocimum_basilicum",

            # Specialty Crops
            "vanilla_planifolia", "avocado_persea_americana", "kiwi_actinidia_deliciosa",
            "fig_ficus_carica", "pomegranate_punica_granatum",

            # Flowers for Pollinator Support
            "sunflower_helianthus_annuus", "marigold_tagetes", "cosmos_cosmos_bipinnatus",
            "zinnia_zinnia_elegans", "nasturtium_tropaeolum",

            # Tree Nuts
            "walnut_juglans_regia", "hazelnut_corylus_avellana", "pecan_carya_illinoinensis",

            # Other Agricultural Flowers
            "buckwheat_fagopyrum_esculentum", "rapeseed_brassica_napus", "canola_brassica_napus"
        ]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for flower detection and analysis.

        Args:
            x: Input image tensor [batch_size, channels, height, width]

        Returns:
            Dictionary containing detection results
        """
        # Get base features from IntelleSwarm EdgePerceptionTransformer
        logits, cls_features = self.base_model(x)  # Returns (logits, cls_feat)

        # Use the class token features directly (already pooled)
        global_features = cls_features  # [batch, emb_dim]

        # Detection outputs
        detections = self.flower_detection_head(global_features)
        species_logits = self.species_classifier(global_features)
        status_logits = self.pollination_status_head(global_features)
        depth = self.depth_estimator(global_features)
        pollen_quality = self.pollen_quality_estimator(global_features)

        return {
            'detections': detections,           # [batch, 5] - bounding box + objectness
            'species_logits': species_logits,  # [batch, num_species]
            'status_logits': status_logits,    # [batch, num_status]
            'depth': depth,                    # [batch, 1] - estimated depth
            'pollen_quality': pollen_quality,  # [batch, 1] - pollen source quality
            'features': cls_features,          # [batch, emb_dim] - class token features
            'base_logits': logits             # [batch, num_classes] - base model logits
        }

    def detect_flowers(self, image: np.ndarray,
                      confidence_threshold: float = 0.5) -> List[FlowerDetection]:
        """
        Detect flowers in a single image.

        Args:
            image: Input image as numpy array (H, W, C)
            confidence_threshold: Minimum confidence for detections

        Returns:
            List of flower detections
        """
        # Preprocess image
        image_tensor = self._preprocess_image(image)

        with torch.no_grad():
            # Forward pass
            outputs = self.forward(image_tensor.unsqueeze(0))

            # Post-process outputs
            detections = self._postprocess_outputs(outputs, image.shape, confidence_threshold)

        return detections

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        # Resize to model input size
        image_resized = cv2.resize(image, self.img_size)

        # Normalize to [0, 1] and convert to tensor
        image_tensor = torch.from_numpy(image_resized).float() / 255.0

        # Rearrange dimensions to (C, H, W)
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.permute(2, 0, 1)

        return image_tensor

    def _postprocess_outputs(self, outputs: Dict[str, torch.Tensor],
                           original_shape: Tuple[int, int, int],
                           confidence_threshold: float) -> List[FlowerDetection]:
        """Post-process model outputs to flower detections."""
        detections = []

        # Extract outputs
        bbox_pred = outputs['detections'][0]  # [5] - first batch item
        species_logits = outputs['species_logits'][0]  # [num_species]
        status_logits = outputs['status_logits'][0]  # [num_status]
        depth = outputs['depth'][0].item()
        pollen_quality = outputs['pollen_quality'][0].item()

        # Check objectness confidence
        objectness = torch.sigmoid(bbox_pred[4]).item()
        if objectness < confidence_threshold:
            return detections

        # Convert bounding box to original image coordinates
        h_orig, w_orig = original_shape[:2]
        h_model, w_model = self.img_size[1], self.img_size[0]

        x_center = torch.sigmoid(bbox_pred[0]).item() * w_model
        y_center = torch.sigmoid(bbox_pred[1]).item() * h_model
        width = torch.sigmoid(bbox_pred[2]).item() * w_model
        height = torch.sigmoid(bbox_pred[3]).item() * h_model

        # Scale to original image size
        x_center = x_center * (w_orig / w_model)
        y_center = y_center * (h_orig / h_model)
        width = width * (w_orig / w_model)
        height = height * (h_orig / h_model)

        # Convert to bbox format
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)

        # Get species prediction
        species_idx = torch.argmax(species_logits).item()
        species = self.species_names[species_idx] if species_idx < len(self.species_names) else "unknown"

        # Get pollination status
        status_idx = torch.argmax(status_logits).item()
        status_values = list(PollinationStatus)
        pollination_status = status_values[status_idx] if status_idx < len(status_values) else PollinationStatus.READY

        # Calculate pollen need urgency based on status
        pollen_need_urgency = self._calculate_pollen_urgency(pollination_status)

        detection = FlowerDetection(
            bbox=(x1, y1, x2, y2),
            species=species,
            pollination_status=pollination_status,
            confidence=objectness,
            center_3d=(x_center, y_center, depth),
            pollen_source_quality=pollen_quality,
            pollen_need_urgency=pollen_need_urgency
        )

        detections.append(detection)
        return detections

    def _calculate_pollen_urgency(self, status: PollinationStatus) -> float:
        """Calculate pollination urgency based on flower status."""
        urgency_map = {
            PollinationStatus.READY: 0.9,        # High urgency - ready for pollination
            PollinationStatus.BUD: 0.2,          # Low urgency - not ready yet
            PollinationStatus.POLLINATED: 0.0,   # No urgency - already done
            PollinationStatus.PAST_PRIME: 0.1,   # Very low - probably too late
            PollinationStatus.DAMAGED: 0.0       # No urgency - can't be pollinated
        }
        return urgency_map.get(status, 0.5)


class PollinationStatusClassifier(nn.Module):
    """
    Specialized classifier for detailed pollination status assessment.

    Provides more granular analysis of flower readiness and timing.
    """

    def __init__(self, input_dim: int = 256, num_classes: int = 5):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        # Time-sensitive features
        self.time_predictor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Hours until optimal pollination window
            nn.ReLU()
        )

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Classify pollination status with timing information.

        Args:
            features: Feature tensor from flower detector

        Returns:
            Classification results with timing
        """
        status_logits = self.classifier(features)
        time_to_optimal = self.time_predictor(features)

        return {
            'status_logits': status_logits,
            'time_to_optimal': time_to_optimal
        }

    def get_pollination_schedule(self, features: torch.Tensor,
                               current_time: float) -> Dict[str, float]:
        """
        Generate pollination scheduling recommendations.

        Args:
            features: Flower features
            current_time: Current timestamp

        Returns:
            Scheduling recommendations
        """
        with torch.no_grad():
            outputs = self.forward(features)
            time_to_optimal = outputs['time_to_optimal'].item()

            # Convert to hours
            optimal_time = current_time + (time_to_optimal * 3600)

            return {
                'optimal_pollination_time': optimal_time,
                'urgency_score': max(0, 1 - (time_to_optimal / 24)),  # Higher urgency as time approaches
                'window_duration_hours': 4.0,  # Typical pollination window
                'confidence': torch.softmax(outputs['status_logits'], dim=-1).max().item()
            }