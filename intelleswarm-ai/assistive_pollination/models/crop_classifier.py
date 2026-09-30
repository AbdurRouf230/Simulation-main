"""
Crop Species Classifier for Agricultural Pollination.

Identifies crop types and their specific pollination requirements
to optimize drone swarm behavior for different agricultural contexts.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json


class PollinationMethod(Enum):
    """Different pollination methods required by crops."""
    CROSS_POLLINATION = "cross"          # Requires pollen from different variety
    SELF_POLLINATION = "self"            # Can self-pollinate but benefits from assistance
    WIND_POLLINATION = "wind"            # Wind-pollinated but can benefit from supplementation
    INSECT_POLLINATION = "insect"        # Requires insect/drone pollination
    MIXED_POLLINATION = "mixed"          # Benefits from multiple methods


class BloomTiming(Enum):
    """Typical bloom timing for scheduling."""
    EARLY_SPRING = "early_spring"        # March-April
    LATE_SPRING = "late_spring"          # April-May
    EARLY_SUMMER = "early_summer"        # May-June
    MID_SUMMER = "mid_summer"            # June-July
    LATE_SUMMER = "late_summer"          # July-August
    FALL = "fall"                        # August-October
    EXTENDED = "extended"                # Multiple bloom periods


@dataclass
class CropSpecification:
    """Detailed crop pollination specifications."""
    name: str
    scientific_name: str
    pollination_method: PollinationMethod
    bloom_timing: BloomTiming
    flower_height_range: Tuple[float, float]  # Height range in meters
    pollen_transfer_method: str              # "brush", "vibration", "electrostatic"
    cross_pollination_distance: float       # Meters for cross-pollination
    optimal_temperature_range: Tuple[float, float]  # Celsius
    optimal_humidity_range: Tuple[float, float]     # Percentage
    bloom_duration_days: int
    flowers_per_plant_estimate: int
    economic_value_per_flower: float  # Estimated economic impact
    pollination_success_indicators: List[str]


class CropSpeciesClassifier(nn.Module):
    """
    Classifier for agricultural crop species and their pollination requirements.

    Extends flower detection to provide crop-specific optimization parameters
    for drone swarm pollination strategies.
    """

    def __init__(self,
                 input_features: int = 256,
                 num_crop_families: int = 15,
                 num_species: int = 50):
        super().__init__()

        self.num_crop_families = num_crop_families
        self.num_species = num_species

        # Hierarchical classification: Family → Species
        self.family_classifier = nn.Sequential(
            nn.Linear(input_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_crop_families)
        )

        self.species_classifier = nn.Sequential(
            nn.Linear(input_features + num_crop_families, 512),  # Features + family prediction
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_species)
        )

        # Pollination requirement predictors
        self.pollination_method_predictor = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(PollinationMethod))
        )

        self.bloom_timing_predictor = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, len(BloomTiming))
        )

        # Quantitative parameter predictors
        self.flower_height_predictor = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # [min_height, max_height]
            nn.Sigmoid()  # Scale to 0-1, will multiply by max_height (10m)
        )

        self.environmental_preferences = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # [temp_min, temp_max, humidity_min, humidity_max]
            nn.Sigmoid()
        )

        # Economic value estimator
        self.economic_value_predictor = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Normalize to 0-1, scale later
        )

        # Load crop specifications database
        self.crop_specs = self._load_crop_specifications()
        self.family_names = list(self.crop_specs.keys())

    def _load_crop_specifications(self) -> Dict[str, Dict[str, CropSpecification]]:
        """Load comprehensive crop specification database."""

        specs = {
            "Rosaceae": {  # Apple, cherry, almond, etc.
                "apple": CropSpecification(
                    name="Apple",
                    scientific_name="Malus domestica",
                    pollination_method=PollinationMethod.CROSS_POLLINATION,
                    bloom_timing=BloomTiming.LATE_SPRING,
                    flower_height_range=(2.0, 6.0),
                    pollen_transfer_method="brush",
                    cross_pollination_distance=10.0,
                    optimal_temperature_range=(15, 25),
                    optimal_humidity_range=(50, 80),
                    bloom_duration_days=10,
                    flowers_per_plant_estimate=50000,
                    economic_value_per_flower=0.002,
                    pollination_success_indicators=["pollen_tube_growth", "fruit_set"]
                ),
                "cherry": CropSpecification(
                    name="Sweet Cherry",
                    scientific_name="Prunus avium",
                    pollination_method=PollinationMethod.CROSS_POLLINATION,
                    bloom_timing=BloomTiming.EARLY_SPRING,
                    flower_height_range=(3.0, 8.0),
                    pollen_transfer_method="brush",
                    cross_pollination_distance=15.0,
                    optimal_temperature_range=(10, 20),
                    optimal_humidity_range=(60, 85),
                    bloom_duration_days=7,
                    flowers_per_plant_estimate=30000,
                    economic_value_per_flower=0.01,
                    pollination_success_indicators=["pollen_adhesion", "stigma_receptivity"]
                ),
                "almond": CropSpecification(
                    name="Almond",
                    scientific_name="Prunus dulcis",
                    pollination_method=PollinationMethod.CROSS_POLLINATION,
                    bloom_timing=BloomTiming.EARLY_SPRING,
                    flower_height_range=(2.5, 7.0),
                    pollen_transfer_method="vibration",
                    cross_pollination_distance=20.0,
                    optimal_temperature_range=(12, 22),
                    optimal_humidity_range=(40, 70),
                    bloom_duration_days=14,
                    flowers_per_plant_estimate=40000,
                    economic_value_per_flower=0.005,
                    pollination_success_indicators=["nut_development", "kernel_formation"]
                )
            },

            "Solanaceae": {  # Tomato, pepper, eggplant
                "tomato": CropSpecification(
                    name="Tomato",
                    scientific_name="Solanum lycopersicum",
                    pollination_method=PollinationMethod.SELF_POLLINATION,
                    bloom_timing=BloomTiming.EXTENDED,
                    flower_height_range=(0.5, 2.5),
                    pollen_transfer_method="vibration",
                    cross_pollination_distance=1.0,
                    optimal_temperature_range=(18, 28),
                    optimal_humidity_range=(60, 80),
                    bloom_duration_days=3,
                    flowers_per_plant_estimate=200,
                    economic_value_per_flower=0.05,
                    pollination_success_indicators=["fruit_development", "seed_set"]
                )
            },

            "Cucurbitaceae": {  # Cucumber, squash, melon
                "cucumber": CropSpecification(
                    name="Cucumber",
                    scientific_name="Cucumis sativus",
                    pollination_method=PollinationMethod.CROSS_POLLINATION,
                    bloom_timing=BloomTiming.MID_SUMMER,
                    flower_height_range=(0.3, 1.5),
                    pollen_transfer_method="brush",
                    cross_pollination_distance=5.0,
                    optimal_temperature_range=(20, 30),
                    optimal_humidity_range=(70, 90),
                    bloom_duration_days=1,
                    flowers_per_plant_estimate=50,
                    economic_value_per_flower=0.2,
                    pollination_success_indicators=["fruit_elongation", "seed_cavity"]
                )
            },

            "Ericaceae": {  # Blueberry, cranberry
                "blueberry": CropSpecification(
                    name="Blueberry",
                    scientific_name="Vaccinium corymbosum",
                    pollination_method=PollinationMethod.CROSS_POLLINATION,
                    bloom_timing=BloomTiming.LATE_SPRING,
                    flower_height_range=(1.0, 3.0),
                    pollen_transfer_method="vibration",
                    cross_pollination_distance=25.0,
                    optimal_temperature_range=(15, 25),
                    optimal_humidity_range=(50, 75),
                    bloom_duration_days=21,
                    flowers_per_plant_estimate=5000,
                    economic_value_per_flower=0.01,
                    pollination_success_indicators=["berry_size", "seed_count"]
                )
            },

            "Fabaceae": {  # Beans, peas, soybeans
                "bean": CropSpecification(
                    name="Common Bean",
                    scientific_name="Phaseolus vulgaris",
                    pollination_method=PollinationMethod.SELF_POLLINATION,
                    bloom_timing=BloomTiming.MID_SUMMER,
                    flower_height_range=(0.2, 1.0),
                    pollen_transfer_method="brush",
                    cross_pollination_distance=1.0,
                    optimal_temperature_range=(18, 25),
                    optimal_humidity_range=(60, 80),
                    bloom_duration_days=2,
                    flowers_per_plant_estimate=100,
                    economic_value_per_flower=0.03,
                    pollination_success_indicators=["pod_formation", "seed_development"]
                )
            }
        }

        return specs

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Classify crop species and predict pollination requirements.

        Args:
            features: Feature tensor from flower detector

        Returns:
            Comprehensive crop classification and requirements
        """
        batch_size = features.size(0)

        # Hierarchical classification
        family_logits = self.family_classifier(features)
        family_probs = torch.softmax(family_logits, dim=-1)

        # Concatenate features with family prediction for species classification
        species_input = torch.cat([features, family_probs], dim=-1)
        species_logits = self.species_classifier(species_input)

        # Pollination requirement predictions
        pollination_method_logits = self.pollination_method_predictor(features)
        bloom_timing_logits = self.bloom_timing_predictor(features)

        # Quantitative predictions
        flower_heights = self.flower_height_predictor(features) * 10.0  # Scale to 0-10m
        env_prefs = self.environmental_preferences(features)

        # Scale environmental preferences
        temp_range = env_prefs[:, :2] * 40 + 5  # 5-45°C
        humidity_range = env_prefs[:, 2:] * 70 + 20  # 20-90%

        economic_value = self.economic_value_predictor(features) * 0.1  # Scale to reasonable range

        return {
            'family_logits': family_logits,
            'species_logits': species_logits,
            'pollination_method_logits': pollination_method_logits,
            'bloom_timing_logits': bloom_timing_logits,
            'flower_heights': flower_heights,
            'temperature_range': temp_range,
            'humidity_range': humidity_range,
            'economic_value': economic_value
        }

    def get_pollination_strategy(self, features: torch.Tensor) -> Dict[str, any]:
        """
        Generate comprehensive pollination strategy for detected crop.

        Args:
            features: Crop features from detection model

        Returns:
            Detailed pollination strategy recommendations
        """
        with torch.no_grad():
            outputs = self.forward(features)

            # Get most likely classifications
            family_idx = torch.argmax(outputs['family_logits'], dim=-1).item()
            species_idx = torch.argmax(outputs['species_logits'], dim=-1).item()
            method_idx = torch.argmax(outputs['pollination_method_logits'], dim=-1).item()
            timing_idx = torch.argmax(outputs['bloom_timing_logits'], dim=-1).item()

            family_name = self.family_names[family_idx] if family_idx < len(self.family_names) else "Unknown"

            # Get crop specifications if available
            crop_specs = None
            if family_name in self.crop_specs:
                species_names = list(self.crop_specs[family_name].keys())
                if species_idx < len(species_names):
                    crop_specs = self.crop_specs[family_name][species_names[species_idx]]

            strategy = {
                'crop_family': family_name,
                'predicted_species': crop_specs.name if crop_specs else "Unknown",
                'scientific_name': crop_specs.scientific_name if crop_specs else "Unknown",
                'pollination_method': list(PollinationMethod)[method_idx].value,
                'bloom_timing': list(BloomTiming)[timing_idx].value,

                # Quantitative parameters
                'flower_height_range': outputs['flower_heights'][0].tolist(),
                'optimal_temperature': outputs['temperature_range'][0].tolist(),
                'optimal_humidity': outputs['humidity_range'][0].tolist(),
                'economic_value_per_flower': outputs['economic_value'][0].item(),

                # Derived strategy parameters
                'recommended_approach_pattern': self._get_approach_pattern(method_idx),
                'pollen_transfer_technique': crop_specs.pollen_transfer_method if crop_specs else "brush",
                'cross_pollination_distance': crop_specs.cross_pollination_distance if crop_specs else 5.0,
                'priority_multiplier': self._calculate_priority_multiplier(outputs['economic_value'][0].item()),

                # Confidence scores
                'classification_confidence': torch.softmax(outputs['species_logits'], dim=-1).max().item(),
                'method_confidence': torch.softmax(outputs['pollination_method_logits'], dim=-1).max().item()
            }

            return strategy

    def _get_approach_pattern(self, method_idx: int) -> str:
        """Determine drone approach pattern based on pollination method."""
        patterns = {
            0: "cross_field_zigzag",    # Cross-pollination
            1: "individual_plant",      # Self-pollination
            2: "wind_assisted",         # Wind-pollination
            3: "systematic_coverage",   # Insect-pollination
            4: "adaptive_mixed"         # Mixed-pollination
        }
        return patterns.get(method_idx, "systematic_coverage")

    def _calculate_priority_multiplier(self, economic_value: float) -> float:
        """Calculate priority multiplier based on economic value."""
        # Higher economic value = higher priority
        return min(3.0, 1.0 + economic_value * 20.0)

    def get_seasonal_schedule(self, location_lat: float,
                            current_month: int) -> List[Dict[str, any]]:
        """
        Generate seasonal pollination schedule for a location.

        Args:
            location_lat: Latitude for seasonal timing adjustment
            current_month: Current month (1-12)

        Returns:
            List of scheduled crop pollination periods
        """
        # Adjust timing based on latitude (simplified model)
        lat_adjustment = (location_lat - 40) * 0.5  # Days adjustment per degree from 40°N

        schedule = []

        for family_name, species_dict in self.crop_specs.items():
            for species_name, spec in species_dict.items():
                # Convert bloom timing to months
                bloom_months = self._timing_to_months(spec.bloom_timing)

                # Adjust for latitude
                adjusted_months = [(m + int(lat_adjustment / 30)) % 12 + 1 for m in bloom_months]

                schedule.append({
                    'crop': spec.name,
                    'family': family_name,
                    'bloom_months': adjusted_months,
                    'duration_days': spec.bloom_duration_days,
                    'active_now': current_month in adjusted_months,
                    'economic_priority': spec.economic_value_per_flower,
                    'pollination_method': spec.pollination_method.value,
                    'flowers_per_plant': spec.flowers_per_plant_estimate
                })

        # Sort by current relevance and economic value
        schedule.sort(key=lambda x: (x['active_now'], x['economic_priority']), reverse=True)

        return schedule

    def _timing_to_months(self, timing: BloomTiming) -> List[int]:
        """Convert bloom timing enum to month numbers."""
        timing_map = {
            BloomTiming.EARLY_SPRING: [3, 4],
            BloomTiming.LATE_SPRING: [4, 5],
            BloomTiming.EARLY_SUMMER: [5, 6],
            BloomTiming.MID_SUMMER: [6, 7],
            BloomTiming.LATE_SUMMER: [7, 8],
            BloomTiming.FALL: [8, 9, 10],
            BloomTiming.EXTENDED: [5, 6, 7, 8, 9]
        }
        return timing_map.get(timing, [6, 7])