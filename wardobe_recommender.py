from datasets import load_dataset
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch
from typing import List, Dict, Any
import random
from dataclasses import dataclass
from PIL import Image
import io
import base64

@dataclass
class FashionItem:
    id: str
    name: str
    category: str
    price: float
    color: str
    style_tags: List[str]
    image_url: str
    purchase_link: str
    description: str
    image_data: str = None  # Base64 encoded image data

class WardrobeRecommender:
    def __init__(self):
        self.dataset = None
        self.item_embeddings = None
        self.categories = {
            'tops': ['shirt', 'blouse', 't-shirt', 'sweater', 'jacket'],
            'bottoms': ['pants', 'jeans', 'skirt', 'shorts'],
            'dresses': ['dress', 'gown', 'jumpsuit'],
            'shoes': ['sneakers', 'heels', 'boots', 'flats'],
            'accessories': ['bag', 'jewelry', 'belt', 'scarf']
        }
        self.occasion_styles = {
            'Wedding': ['Formal', 'Elegant', 'Classic'],
            'Business Meeting': ['Professional', 'Formal', 'Conservative'],
            'Casual Outing': ['Casual', 'Comfortable', 'Streetwear'],
            'Party': ['Trendy', 'Stylish', 'Bold'],
            'Date Night': ['Elegant', 'Romantic', 'Stylish']
        }
        self.load_dataset()
    
    def load_dataset(self):
        """Load and prepare the Polyvore dataset"""
        try:
            self.dataset = load_dataset("Marqo/polyvore", split='train')
            print(f"Loaded dataset with {len(self.dataset)} items")
        except Exception as e:
            print(f"Warning: Could not load dataset: {str(e)}")
            # Initialize with sample data for testing
            self._initialize_sample_data()
    
    def _initialize_sample_data(self):
        """Initialize sample data for testing when dataset is not available"""
        self.sample_items = {
            'tops': [
                FashionItem(
                    id="t1",
                    name="Classic White Button-Down Shirt",
                    category="tops",
                    price=49.99,
                    color="White",
                    style_tags=["Classic", "Formal", "Professional"],
                    image_url="sample_url",
                    purchase_link="https://example.com/shirt1",
                    description="Versatile white cotton button-down shirt"
                ),
                # Add more sample items...
            ],
            'bottoms': [
                FashionItem(
                    id="b1",
                    name="Black Tailored Pants",
                    category="bottoms",
                    price=79.99,
                    color="Black",
                    style_tags=["Professional", "Formal", "Classic"],
                    image_url="sample_url",
                    purchase_link="https://example.com/pants1",
                    description="Classic black tailored trousers"
                ),
                # Add more sample items...
            ],
            # Add more categories...
        }
    
    def get_outfit_recommendations(
        self,
        style_profile: torch.Tensor,
        occasion: str,
        budget: float,
        preferences: Dict[str, Any],
        num_recommendations: int = 3
    ) -> List[Dict[str, Any]]:
        """Generate outfit recommendations"""
        try:
            if self.dataset is not None:
                return self._get_recommendations_from_dataset(
                    style_profile, occasion, budget, preferences, num_recommendations
                )
            else:
                return self._get_recommendations_from_samples(
                    occasion, budget, preferences, num_recommendations
                )
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return self._get_fallback_recommendations(budget, num_recommendations)
    
    def _get_recommendations_from_dataset(
        self,
        style_profile: torch.Tensor,
        occasion: str,
        budget: float,
        preferences: Dict[str, Any],
        num_recommendations: int
    ) -> List[Dict[str, Any]]:
        """Generate recommendations using the Polyvore dataset"""
        recommendations = []
        
        # Get style tags for the occasion
        occasion_styles = self.occasion_styles.get(occasion, ['Casual'])
        
        # Filter items by occasion and preferences
        filtered_items = []
        for item in self.dataset:
            if (
                item.get('price', float('inf')) <= budget * 0.4  # Single item should not exceed 40% of budget
                and self._matches_preferences(item, preferences)
                and any(style in item.get('style_tags', []) for style in occasion_styles)
            ):
                # Extract image data if available
                image_data = item.get('image', None)
                if image_data:
                    try:
                        # Convert image data to base64 if it's not already
                        if isinstance(image_data, bytes):
                            image_data = base64.b64encode(image_data).decode('utf-8')
                        item['image_data'] = image_data
                    except Exception as e:
                        print(f"Error processing image data: {e}")
                        item['image_data'] = None
                filtered_items.append(item)
        
        # Create outfits from filtered items
        for _ in range(num_recommendations):
            outfit = self._create_outfit(filtered_items, budget)
            if outfit:
                recommendations.append(outfit)
        
        return recommendations
    
    def _get_recommendations_from_samples(
        self,
        occasion: str,
        budget: float,
        preferences: Dict[str, Any],
        num_recommendations: int
    ) -> List[Dict[str, Any]]:
        """Generate recommendations using sample data"""
        recommendations = []
        
        for i in range(num_recommendations):
            total_price = 0
            outfit_items = []
            
            # Add one item from each necessary category
            for category in ['tops', 'bottoms', 'shoes']:
                if category in self.sample_items:
                    items = self.sample_items[category]
                    if items:
                        item = random.choice(items)
                        if total_price + item.price <= budget:
                            outfit_items.append({
                                'name': item.name,
                                'category': item.category,
                                'price': item.price,
                                'color': item.color,
                                'purchase_link': item.purchase_link,
                                'description': item.description
                            })
                            total_price += item.price
            
            if outfit_items:
                recommendations.append({
                    'set_id': f'outfit_{i+1}',
                    'total_price': total_price,
                    'items': outfit_items
                })
        
        return recommendations
    
    def _get_fallback_recommendations(
        self,
        budget: float,
        num_recommendations: int
    ) -> List[Dict[str, Any]]:
        """Generate fallback recommendations when other methods fail"""
        recommendations = []
        
        for i in range(num_recommendations):
            total_price = budget * random.uniform(0.7, 0.95)
            outfit = {
                'set_id': f'outfit_{i+1}',
                'total_price': total_price,
                'items': [
                    {
                        'name': 'Classic Shirt',
                        'category': 'tops',
                        'price': total_price * 0.3,
                        'color': 'White',
                        'purchase_link': 'https://example.com/shirt',
                        'description': 'A versatile classic shirt'
                    },
                    {
                        'name': 'Tailored Pants',
                        'category': 'bottoms',
                        'price': total_price * 0.4,
                        'color': 'Black',
                        'purchase_link': 'https://example.com/pants',
                        'description': 'Professional tailored pants'
                    },
                    {
                        'name': 'Classic Shoes',
                        'category': 'shoes',
                        'price': total_price * 0.3,
                        'color': 'Black',
                        'purchase_link': 'https://example.com/shoes',
                        'description': 'Comfortable classic shoes'
                    }
                ]
            }
            recommendations.append(outfit)
        
        return recommendations
    
    def _matches_preferences(self, item: Dict, preferences: Dict) -> bool:
        """Check if an item matches user preferences"""
        if not preferences:
            return True
            
        color_prefs = preferences.get('color_preference', [])
        style_prefs = preferences.get('style_preference', [])
        
        color_match = not color_prefs or item.get('color') in color_prefs
        style_match = not style_prefs or any(
            style in item.get('style_tags', []) for style in style_prefs
        )
        
        return color_match and style_match
    
    def _create_outfit(self, items: List[Dict], budget: float) -> Dict[str, Any]:
        """Create a complete outfit from available items within budget"""
        outfit_items = []
        total_price = 0
        
        # Ensure we have one item from each main category
        for category in ['tops', 'bottoms', 'shoes']:
            category_items = [
                item for item in items 
                if item.get('category', '').lower() == category 
                and total_price + item.get('price', 0) <= budget
            ]
            
            if category_items:
                selected_item = random.choice(category_items)
                outfit_items.append({
                    'name': selected_item.get('name', f'Item {len(outfit_items) + 1}'),
                    'category': selected_item.get('category', '').lower(),
                    'price': selected_item.get('price', 0),
                    'color': selected_item.get('color', 'Unknown'),
                    'purchase_link': selected_item.get('purchase_link', 'https://example.com'),
                    'description': selected_item.get('description', ''),
                    'image_data': selected_item.get('image_data')
                })
                total_price += selected_item.get('price', 0)
        
        if len(outfit_items) < 3:  # If we couldn't get all necessary items
            return None
            
        return {
            'set_id': f'outfit_{random.randint(1000, 9999)}',
            'total_price': total_price,
            'items': outfit_items
        }
    
    def filter_by_budget(self, items: List[Dict], budget: float) -> List[Dict]:
        """Filter items by budget constraints"""
        filtered_items = []
        total_cost = 0
        
        for item in items:
            if total_cost + item['price'] <= budget:
                filtered_items.append(item)
                total_cost += item['price']
                
        return filtered_items
    
    def filter_by_preferences(
        self,
        items: List[Dict],
        color_preferences: List[str],
        style_preferences: List[str]
    ) -> List[Dict]:
        """Filter items by user preferences"""
        if not color_preferences and not style_preferences:
            return items
            
        filtered_items = []
        for item in items:
            color_match = not color_preferences or item['color'] in color_preferences
            style_match = not style_preferences or any(style in item.get('style_tags', []) for style in style_preferences)
            
            if color_match and style_match:
                filtered_items.append(item)
                
        return filtered_items 
