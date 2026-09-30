"""
Environmental Condition Analyzer for Agricultural Pollination.

Analyzes weather, lighting, and environmental conditions to optimize
drone pollination timing and techniques.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime, timedelta


class WeatherCondition(Enum):
    """Weather condition categories for pollination."""
    OPTIMAL = "optimal"          # Perfect conditions
    GOOD = "good"               # Good conditions
    MARGINAL = "marginal"       # Challenging but workable
    POOR = "poor"              # Difficult conditions
    DANGEROUS = "dangerous"     # Unsafe for drone operation


class PollinationViability(Enum):
    """Pollination success likelihood under current conditions."""
    EXCELLENT = "excellent"     # >90% success rate expected
    GOOD = "good"              # 70-90% success rate
    MODERATE = "moderate"       # 50-70% success rate
    LOW = "low"                # 20-50% success rate
    VERY_LOW = "very_low"      # <20% success rate


@dataclass
class EnvironmentalConditions:
    """Current environmental conditions."""
    temperature_c: float
    humidity_percent: float
    wind_speed_ms: float
    wind_direction_deg: float
    pressure_hpa: float
    light_intensity_lux: float
    precipitation_mm_h: float
    cloud_cover_percent: float
    uv_index: float
    air_quality_aqi: Optional[int] = None


@dataclass
class PollinationForecast:
    """Pollination condition forecast."""
    timestamp: float
    weather_condition: WeatherCondition
    pollination_viability: PollinationViability
    recommended_actions: List[str]
    drone_safety_score: float  # 0-1, where 1 is completely safe
    pollen_transfer_efficiency: float  # Expected efficiency 0-1
    optimal_flight_altitude: float  # Recommended altitude in meters
    wind_compensation_required: bool


class EnvironmentalConditionAnalyzer(nn.Module):
    """
    Neural network for analyzing environmental conditions and their impact
    on pollination success and drone operations.
    """

    def __init__(self, input_features: int = 16):
        super().__init__()

        self.input_features = input_features

        # Environmental condition encoder
        self.condition_encoder = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )

        # Weather condition classifier
        self.weather_classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(WeatherCondition))
        )

        # Pollination viability predictor
        self.viability_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, len(PollinationViability))
        )

        # Drone safety score predictor
        self.safety_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 0-1 safety score
        )

        # Pollen transfer efficiency predictor
        self.efficiency_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 0-1 efficiency score
        )

        # Optimal flight altitude predictor
        self.altitude_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()  # Non-negative altitude
        )

        # Time-series forecasting for conditions
        self.lstm_forecaster = nn.LSTM(
            input_size=input_features,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )

        self.forecast_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, input_features)  # Predict future conditions
        )

    def forward(self, environmental_data: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Analyze environmental conditions.

        Args:
            environmental_data: Environmental features tensor [batch_size, features]

        Returns:
            Analysis results dictionary
        """
        # Encode environmental conditions
        encoded = self.condition_encoder(environmental_data)

        # Make predictions
        weather_logits = self.weather_classifier(encoded)
        viability_logits = self.viability_predictor(encoded)
        safety_score = self.safety_predictor(encoded)
        efficiency_score = self.efficiency_predictor(encoded)
        optimal_altitude = self.altitude_predictor(encoded) * 100  # Scale to reasonable range (0-100m)

        return {
            'weather_logits': weather_logits,
            'viability_logits': viability_logits,
            'safety_score': safety_score,
            'efficiency_score': efficiency_score,
            'optimal_altitude': optimal_altitude,
            'encoded_features': encoded
        }

    def predict_conditions(self, conditions: EnvironmentalConditions) -> PollinationForecast:
        """
        Predict pollination conditions from environmental data.

        Args:
            conditions: Current environmental conditions

        Returns:
            Pollination forecast
        """
        # Convert conditions to tensor
        features = self._conditions_to_tensor(conditions)

        with torch.no_grad():
            outputs = self.forward(features.unsqueeze(0))

            # Get predictions
            weather_idx = torch.argmax(outputs['weather_logits']).item()
            viability_idx = torch.argmax(outputs['viability_logits']).item()

            weather_condition = list(WeatherCondition)[weather_idx]
            pollination_viability = list(PollinationViability)[viability_idx]

            safety_score = outputs['safety_score'].item()
            efficiency_score = outputs['efficiency_score'].item()
            optimal_altitude = outputs['optimal_altitude'].item()

            # Generate recommendations
            recommendations = self._generate_recommendations(
                conditions, weather_condition, pollination_viability, safety_score
            )

            forecast = PollinationForecast(
                timestamp=time.time(),
                weather_condition=weather_condition,
                pollination_viability=pollination_viability,
                recommended_actions=recommendations,
                drone_safety_score=safety_score,
                pollen_transfer_efficiency=efficiency_score,
                optimal_flight_altitude=optimal_altitude,
                wind_compensation_required=conditions.wind_speed_ms > 3.0
            )

            return forecast

    def forecast_conditions(self,
                          historical_conditions: List[EnvironmentalConditions],
                          forecast_hours: int = 24) -> List[PollinationForecast]:
        """
        Forecast pollination conditions for the next several hours.

        Args:
            historical_conditions: Recent environmental data
            forecast_hours: Hours to forecast ahead

        Returns:
            List of hourly forecasts
        """
        if len(historical_conditions) < 12:  # Need at least 12 hours of data
            return []

        # Convert to tensor sequence
        sequence = torch.stack([
            self._conditions_to_tensor(cond) for cond in historical_conditions[-12:]
        ]).unsqueeze(0)  # [1, sequence_length, features]

        forecasts = []

        with torch.no_grad():
            # Initialize LSTM state
            h_0 = torch.zeros(2, 1, 64)  # [num_layers, batch_size, hidden_size]
            c_0 = torch.zeros(2, 1, 64)

            # Process historical data
            lstm_out, (h_n, c_n) = self.lstm_forecaster(sequence, (h_0, c_0))

            # Generate forecasts
            last_output = lstm_out[:, -1:, :]  # Last time step

            for hour in range(forecast_hours):
                # Predict next conditions
                predicted_features = self.forecast_head(last_output)

                # Convert back to conditions (simplified)
                predicted_conditions = self._tensor_to_conditions(predicted_features.squeeze())

                # Get forecast
                forecast = self.predict_conditions(predicted_conditions)
                forecast.timestamp = time.time() + (hour + 1) * 3600  # Hours ahead

                forecasts.append(forecast)

                # Update for next iteration
                lstm_out, (h_n, c_n) = self.lstm_forecaster(predicted_features, (h_n, c_n))
                last_output = lstm_out

        return forecasts

    def _conditions_to_tensor(self, conditions: EnvironmentalConditions) -> torch.Tensor:
        """Convert environmental conditions to model input tensor."""
        # Normalize features to reasonable ranges
        features = [
            conditions.temperature_c / 40.0,                    # 0-40°C → 0-1
            conditions.humidity_percent / 100.0,               # 0-100% → 0-1
            conditions.wind_speed_ms / 20.0,                   # 0-20 m/s → 0-1
            conditions.wind_direction_deg / 360.0,             # 0-360° → 0-1
            (conditions.pressure_hpa - 900) / 200.0,          # 900-1100 hPa → 0-1
            conditions.light_intensity_lux / 100000.0,         # 0-100k lux → 0-1
            conditions.precipitation_mm_h / 50.0,              # 0-50 mm/h → 0-1
            conditions.cloud_cover_percent / 100.0,            # 0-100% → 0-1
            conditions.uv_index / 12.0,                        # 0-12 → 0-1

            # Time-based features
            datetime.now().hour / 24.0,                        # Hour of day
            datetime.now().timetuple().tm_yday / 365.0,        # Day of year

            # Derived features
            1.0 if 10 <= datetime.now().hour <= 16 else 0.0,   # Daytime pollination window
            1.0 if conditions.temperature_c > 15 else 0.0,     # Temperature threshold
            1.0 if conditions.wind_speed_ms < 5.0 else 0.0,    # Wind threshold
            1.0 if conditions.precipitation_mm_h == 0 else 0.0, # No rain
            1.0 if conditions.humidity_percent < 85 else 0.0    # Humidity threshold
        ]

        return torch.tensor(features, dtype=torch.float32)

    def _tensor_to_conditions(self, tensor: torch.Tensor) -> EnvironmentalConditions:
        """Convert model tensor back to environmental conditions."""
        values = tensor.cpu().numpy()

        return EnvironmentalConditions(
            temperature_c=values[0] * 40.0,
            humidity_percent=values[1] * 100.0,
            wind_speed_ms=values[2] * 20.0,
            wind_direction_deg=values[3] * 360.0,
            pressure_hpa=values[4] * 200.0 + 900,
            light_intensity_lux=values[5] * 100000.0,
            precipitation_mm_h=values[6] * 50.0,
            cloud_cover_percent=values[7] * 100.0,
            uv_index=values[8] * 12.0
        )

    def _generate_recommendations(self,
                                conditions: EnvironmentalConditions,
                                weather: WeatherCondition,
                                viability: PollinationViability,
                                safety_score: float) -> List[str]:
        """Generate actionable recommendations based on conditions."""
        recommendations = []

        # Safety recommendations
        if safety_score < 0.3:
            recommendations.append("ABORT_MISSION - Unsafe conditions for drone operation")
            return recommendations
        elif safety_score < 0.6:
            recommendations.append("CAUTION - Heightened safety protocols required")

        # Wind recommendations
        if conditions.wind_speed_ms > 8.0:
            recommendations.append("HIGH_WIND - Reduce flight speed and increase stability margins")
        elif conditions.wind_speed_ms > 5.0:
            recommendations.append("MODERATE_WIND - Enable enhanced wind compensation")

        # Temperature recommendations
        if conditions.temperature_c < 10:
            recommendations.append("LOW_TEMP - Delay until warmer conditions for better pollen viability")
        elif conditions.temperature_c > 30:
            recommendations.append("HIGH_TEMP - Prioritize morning operations, monitor system cooling")

        # Humidity recommendations
        if conditions.humidity_percent > 85:
            recommendations.append("HIGH_HUMIDITY - Risk of pollen clumping, adjust transfer technique")
        elif conditions.humidity_percent < 30:
            recommendations.append("LOW_HUMIDITY - Pollen may desiccate quickly, increase transfer speed")

        # Rain recommendations
        if conditions.precipitation_mm_h > 0.5:
            recommendations.append("PRECIPITATION - Abort outdoor operations, pollen will be washed away")
        elif conditions.precipitation_mm_h > 0.1:
            recommendations.append("LIGHT_RAIN - Monitor conditions, be ready to abort")

        # Light recommendations
        if conditions.light_intensity_lux < 1000:
            recommendations.append("LOW_LIGHT - Enable enhanced vision systems, reduce flight speed")

        # Timing recommendations
        current_hour = datetime.now().hour
        if current_hour < 8:
            recommendations.append("EARLY_MORNING - Wait for flowers to open and dew to evaporate")
        elif current_hour > 18:
            recommendations.append("EVENING - Flowers may be closing, prioritize urgent pollination only")

        # Viability-based recommendations
        if viability == PollinationViability.EXCELLENT:
            recommendations.append("OPTIMAL_CONDITIONS - Full speed operations recommended")
        elif viability == PollinationViability.LOW:
            recommendations.append("SUBOPTIMAL - Consider postponing non-critical pollination")

        # Default recommendations if conditions are good
        if not recommendations and weather in [WeatherCondition.OPTIMAL, WeatherCondition.GOOD]:
            recommendations.append("PROCEED - Conditions favorable for pollination operations")

        return recommendations

    def get_daily_schedule_optimization(self,
                                      forecast_24h: List[PollinationForecast],
                                      crop_priorities: Dict[str, float]) -> Dict[str, any]:
        """
        Optimize daily pollination schedule based on forecast.

        Args:
            forecast_24h: 24-hour forecast
            crop_priorities: Crop priority weights

        Returns:
            Optimized schedule recommendations
        """
        # Find optimal time windows
        optimal_windows = []
        current_window = None

        for i, forecast in enumerate(forecast_24h):
            is_good = (
                forecast.weather_condition in [WeatherCondition.OPTIMAL, WeatherCondition.GOOD] and
                forecast.pollination_viability in [PollinationViability.EXCELLENT, PollinationViability.GOOD] and
                forecast.drone_safety_score > 0.7
            )

            if is_good:
                if current_window is None:
                    current_window = {
                        'start_hour': i,
                        'end_hour': i,
                        'avg_efficiency': forecast.pollen_transfer_efficiency,
                        'avg_safety': forecast.drone_safety_score
                    }
                else:
                    current_window['end_hour'] = i
                    current_window['avg_efficiency'] = (
                        current_window['avg_efficiency'] + forecast.pollen_transfer_efficiency
                    ) / 2
                    current_window['avg_safety'] = (
                        current_window['avg_safety'] + forecast.drone_safety_score
                    ) / 2
            else:
                if current_window is not None:
                    current_window['duration'] = current_window['end_hour'] - current_window['start_hour'] + 1
                    optimal_windows.append(current_window)
                    current_window = None

        # Close final window if needed
        if current_window is not None:
            current_window['duration'] = current_window['end_hour'] - current_window['start_hour'] + 1
            optimal_windows.append(current_window)

        # Rank windows by quality score
        for window in optimal_windows:
            window['quality_score'] = (
                window['avg_efficiency'] * 0.4 +
                window['avg_safety'] * 0.3 +
                min(window['duration'] / 4.0, 1.0) * 0.3  # Prefer longer windows, cap at 4 hours
            )

        optimal_windows.sort(key=lambda x: x['quality_score'], reverse=True)

        return {
            'optimal_windows': optimal_windows[:3],  # Top 3 windows
            'best_start_time': optimal_windows[0]['start_hour'] if optimal_windows else None,
            'total_good_hours': sum(w['duration'] for w in optimal_windows),
            'recommended_mission_duration': optimal_windows[0]['duration'] if optimal_windows else 0,
            'weather_risk_periods': [
                i for i, f in enumerate(forecast_24h)
                if f.weather_condition == WeatherCondition.DANGEROUS
            ]
        }