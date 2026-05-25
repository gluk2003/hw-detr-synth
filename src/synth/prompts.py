"""Промпты Stable-Diffusion по классам.

Для каждого редкого класса — несколько вариантов positive-промпта
(разные сцены / освещение / погода), общий negative — чтобы избежать
артефактов и расово-смещённого датасета.
"""
from __future__ import annotations

PROMPTS: dict[str, list[str]] = {
    "stop_sign": [
        "a red octagonal stop sign at a city intersection, daytime, "
        "realistic photo, high detail, urban background",
        "a stop sign at a rural crossroad, sunset, golden hour, "
        "documentary photo",
        "a stop sign covered in snow, winter street, soft daylight, photorealistic",
        "a stop sign next to a tree-lined street, rainy weather, wet asphalt, "
        "cinematic lighting",
        "stop sign at night under a streetlight, slight motion blur, "
        "photorealistic, 35mm lens",
    ],
    "traffic_light": [
        "a traffic light hanging above a busy street, daytime, "
        "high resolution photo, urban scene",
        "pedestrian traffic light at a crosswalk, evening, soft lighting, "
        "photorealistic",
        "a traffic light at a city intersection, fog, atmospheric, "
        "cinematic photo",
        "traffic light pole on a suburban road, sunny day, blue sky, "
        "realistic, sharp focus",
        "traffic light glowing red against dark sky, night, raindrops, "
        "photorealistic, telephoto",
    ],
    "motorcycle": [
        "a parked sport motorcycle on a city street, daytime, photorealistic, "
        "ultra-detailed, 50mm lens",
        "a cruiser motorcycle on a country road, sunset, warm light, "
        "documentary photo",
        "an off-road motorcycle on a dirt trail, soft daylight, dust, "
        "photorealistic",
        "a vintage motorcycle in a garage, soft window light, realistic photo, "
        "high detail",
        "a motorcycle parked in front of a coffee shop, urban scene, autumn, "
        "photorealistic",
    ],
}

NEGATIVE_PROMPT = (
    "low quality, blurry, distorted, cartoon, painting, illustration, render, "
    "cgi, low resolution, watermark, text, signature, deformed, mutated, "
    "out of focus, oversaturated, jpeg artifacts"
)
