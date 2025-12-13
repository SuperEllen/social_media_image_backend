import math
import random
from typing import Dict, Any


def predict_engagement(features: Dict[str, Any], user_info: Dict[str, Any]) -> Dict[str, Any]:
    
    # Linear model weights (simplified)
    weights_likes = {
        "Intercept": 4.7536,
        "Low_Level_Attention_Entropy": 0.9208,
        "Mid_Level_Attention_Entropy": 0.9580,
        "High_Level_Attention_Entropy": -1.0318,
        "num_of_pics": 0.0,
        "number_of_tags": -0.0309,
        "number_of_mentions": 0.0,
        "follower_base": 2.245e-06,
        "followees": 0.0003,
        "number_of_posts": -6.48e-05,
        "Warm_Hue_Proportion": 0.0,
        "average_saturation": 0.0,
        "average_brightness": -0.5136,
        "contrast_of_brightness": -0.8345,
        "proportion_brightness": 0.0
    }

    weights_comments = {
        "Intercept": -0.2665,
        "Low_Level_Attention_Entropy": 1.0237,
        "Mid_Level_Attention_Entropy": 0.8027,
        "High_Level_Attention_Entropy": -0.7329,
        "num_of_pics": 0.0286,
        "number_of_tags": -0.0131,
        "number_of_mentions": 0.0156,
        "follower_base": 9.99e-07,
        "followees": -4.64e-05,
        "number_of_posts": -5.3e-05,
        "Warm_Hue_Proportion": 0.1473,
        "average_saturation": -0.3264,
        "average_brightness": 1.5914,
        "contrast_of_brightness": -1.4446,
        "proportion_brightness": -0.8369,
    }

    # Extract and normalize features
    Warm_Hue_Proportion = features.get("Warm_Hue_Proportion", 0.0)
    average_saturation = features.get("Average_Saturation", 0.0)
    average_brightness = features.get("Average_Brightness", 0.5)
    contrast_of_brightness = features.get("Contrast_of_Brightness", 0.5)
    proportion_brightness = features.get("Proportion_Brightness", 0.5)
    Low_Level_Attention_Entropy = features.get("Low_Level_Attention_Entropy", 0.5)
    Mid_Level_Attention_Entropy = features.get("Mid_Level_Attention_Entropy", 0.5)
    High_Level_Attention_Entropy = features.get("High_Level_Attention_Entropy", 0.5)

    
    # Extract user info
    follower_base = user_info.get("follower_base", 100)
    followees = user_info.get("followees", 50)
    number_of_posts = user_info.get("number_of_posts", 50)
    number_of_tags = user_info.get("number_of_tags", 5)
    number_of_mentions = user_info.get("number_of_mentions", 2) 

    predicted_likes = (weights_likes["Intercept"] +
                      weights_likes["Low_Level_Attention_Entropy"] * Low_Level_Attention_Entropy +
                        weights_likes["Mid_Level_Attention_Entropy"] * Mid_Level_Attention_Entropy +
                        weights_likes["High_Level_Attention_Entropy"] * High_Level_Attention_Entropy +
                        weights_likes["num_of_pics"] * 1 +
                        weights_likes["number_of_tags"] * number_of_tags +
                        weights_likes["number_of_mentions"] * number_of_mentions +
                        weights_likes["follower_base"] * follower_base +
                        weights_likes["followees"] * followees +
                        weights_likes["number_of_posts"] * number_of_posts +
                        weights_likes["Warm_Hue_Proportion"] * Warm_Hue_Proportion +
                        weights_likes["average_saturation"] * average_saturation +
                        weights_likes["average_brightness"] * average_brightness +
                        weights_likes["contrast_of_brightness"] * contrast_of_brightness +
                        weights_likes["proportion_brightness"] * proportion_brightness)
    
    predicted_comments = (weights_comments["Intercept"] +
                      weights_comments["Low_Level_Attention_Entropy"] * Low_Level_Attention_Entropy +
                        weights_comments["Mid_Level_Attention_Entropy"] * Mid_Level_Attention_Entropy +
                        weights_comments["High_Level_Attention_Entropy"] * High_Level_Attention_Entropy +
                        weights_comments["num_of_pics"] * 1 +
                        weights_comments["number_of_tags"] * number_of_tags +
                        weights_comments["number_of_mentions"] * number_of_mentions +
                        weights_comments["follower_base"] * follower_base +
                        weights_comments["followees"] * followees +
                        weights_comments["number_of_posts"] * number_of_posts +
                        weights_comments["Warm_Hue_Proportion"] * Warm_Hue_Proportion +
                        weights_comments["average_saturation"] * average_saturation +
                        weights_comments["average_brightness"] * average_brightness +
                        weights_comments["contrast_of_brightness"] * contrast_of_brightness +
                        weights_comments["proportion_brightness"] * proportion_brightness)

    
    return {
        "likes": predicted_likes,
        "comments": predicted_comments
    }