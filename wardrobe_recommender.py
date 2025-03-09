from datasets import load_dataset
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from typing import List, Dict, Any, Tuple
import random
from dataclasses import dataclass
from PIL import Image
import io
import base64
import re
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

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
        self.bm25 = None
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        self.item_vectors = None
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
        self._initialize_search_indices()
    
    def load_dataset(self):
        """Load and prepare the dataset"""
        try:
            # Initialize with sample data since we can't load the Polyvore dataset
            self._initialize_sample_data()
            print("Initialized with sample data")
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            self.dataset = []
            self.sample_items = {}
    
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
                FashionItem(
                    id="t2",
                    name="Navy Blue Blazer",
                    category="tops",
                    price=129.99,
                    color="Navy",
                    style_tags=["Professional", "Formal", "Classic"],
                    image_url="sample_url",
                    purchase_link="https://example.com/blazer1",
                    description="Tailored navy blue blazer perfect for business meetings"
                ),
                FashionItem(
                    id="t3",
                    name="Casual Gray T-Shirt",
                    category="tops",
                    price=24.99,
                    color="Gray",
                    style_tags=["Casual", "Comfortable", "Streetwear"],
                    image_url="sample_url",
                    purchase_link="https://example.com/tshirt1",
                    description="Comfortable cotton t-shirt for everyday wear"
                ),
                FashionItem(
                    id="t4",
                    name="Sequin Party Top",
                    category="tops",
                    price=79.99,
                    color="Gold",
                    style_tags=["Party", "Trendy", "Bold"],
                    image_url="sample_url",
                    purchase_link="https://example.com/top1",
                    description="Sparkly sequin top perfect for parties"
                ),
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
                FashionItem(
                    id="b2",
                    name="Dark Wash Jeans",
                    category="bottoms",
                    price=59.99,
                    color="Blue",
                    style_tags=["Casual", "Versatile", "Streetwear"],
                    image_url="sample_url",
                    purchase_link="https://example.com/jeans1",
                    description="Comfortable dark wash denim jeans"
                ),
                FashionItem(
                    id="b3",
                    name="Leather Mini Skirt",
                    category="bottoms",
                    price=89.99,
                    color="Black",
                    style_tags=["Party", "Trendy", "Bold"],
                    image_url="sample_url",
                    purchase_link="https://example.com/skirt1",
                    description="Edgy leather mini skirt"
                ),
            ],
            'shoes': [
                FashionItem(
                    id="s1",
                    name="Black Leather Oxford Shoes",
                    category="shoes",
                    price=129.99,
                    color="Black",
                    style_tags=["Professional", "Formal", "Classic"],
                    image_url="sample_url",
                    purchase_link="https://example.com/shoes1",
                    description="Classic leather oxford shoes"
                ),
                FashionItem(
                    id="s2",
                    name="White Sneakers",
                    category="shoes",
                    price=69.99,
                    color="White",
                    style_tags=["Casual", "Comfortable", "Streetwear"],
                    image_url="sample_url",
                    purchase_link="https://example.com/sneakers1",
                    description="Clean and minimal white sneakers"
                ),
                FashionItem(
                    id="s3",
                    name="Metallic High Heels",
                    category="shoes",
                    price=99.99,
                    color="Gold",
                    style_tags=["Party", "Trendy", "Bold"],
                    image_url="sample_url",
                    purchase_link="https://example.com/heels1",
                    description="Stunning metallic high heel sandals"
                ),
            ],
            'accessories': [
                FashionItem(
                    id="a1",
                    name="Leather Briefcase",
                    category="accessories",
                    price=199.99,
                    color="Brown",
                    style_tags=["Professional", "Classic", "Formal"],
                    image_url="sample_url",
                    purchase_link="https://example.com/bag1",
                    description="Professional leather briefcase"
                ),
                FashionItem(
                    id="a2",
                    name="Canvas Backpack",
                    category="accessories",
                    price=45.99,
                    color="Gray",
                    style_tags=["Casual", "Practical", "Streetwear"],
                    image_url="sample_url",
                    purchase_link="https://example.com/backpack1",
                    description="Durable canvas backpack for everyday use"
                ),
                FashionItem(
                    id="a3",
                    name="Crystal Statement Necklace",
                    category="accessories",
                    price=49.99,
                    color="Silver",
                    style_tags=["Party", "Bold", "Trendy"],
                    image_url="sample_url",
                    purchase_link="https://example.com/necklace1",
                    description="Eye-catching crystal statement necklace"
                ),
            ],
        }
        
        # Convert sample items to dataset format for BM25 and TF-IDF
        self.dataset = []
        for category, items in self.sample_items.items():
            for item in items:
                self.dataset.append({
                    'id': item.id,
                    'name': item.name,
                    'category': item.category,
                    'price': item.price,
                    'color': item.color,
                    'style_tags': item.style_tags,
                    'description': item.description,
                    'image_url': item.image_url,
                    'purchase_link': item.purchase_link,
                })
        
        # Initialize search indices with sample data
        self._initialize_search_indices()
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for search"""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Tokenize and remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in stop_words]
        return ' '.join(tokens)

    def _initialize_search_indices(self):
        """Initialize BM25 and TF-IDF indices"""
        try:
            if not self.dataset:
                return

            # Prepare documents for BM25 and TF-IDF
            documents = []
            for item in self.dataset:
                # Combine all text fields for better matching
                doc = f"{item.get('name', '')} {item.get('description', '')} {' '.join(item.get('style_tags', []))} {item.get('color', '')} {item.get('category', '')}"
                doc = self._preprocess_text(doc)
                documents.append(doc)

            if not documents:
                return

            # Initialize BM25
            tokenized_documents = [doc.split() for doc in documents]
            self.bm25 = BM25Okapi(tokenized_documents)

            # Initialize TF-IDF
            self.item_vectors = self.tfidf_vectorizer.fit_transform(documents)
            
            print("Search indices initialized successfully")
        except Exception as e:
            print(f"Error initializing search indices: {str(e)}")
            self.bm25 = None
            self.item_vectors = None

    def _get_similar_items(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Get similar items using BM25 and TF-IDF"""
        try:
            if not self.bm25 or not self.item_vectors:
                return []

            # BM25 scoring
            query = self._preprocess_text(query)
            bm25_scores = self.bm25.get_scores(query.split())

            # TF-IDF similarity
            query_vector = self.tfidf_vectorizer.transform([query])
            tfidf_scores = cosine_similarity(query_vector, self.item_vectors)[0]

            # Combine scores (normalized)
            bm25_scores = np.array(bm25_scores)
            tfidf_scores = np.array(tfidf_scores)
            
            # Normalize scores
            bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
            tfidf_scores = (tfidf_scores - tfidf_scores.min()) / (tfidf_scores.max() - tfidf_scores.min() + 1e-8)
            
            # Combine scores
            combined_scores = 0.5 * bm25_scores + 0.5 * tfidf_scores

            # Get top-k items
            top_indices = np.argsort(combined_scores)[-top_k:][::-1]
            return [(int(idx), float(combined_scores[idx])) for idx in top_indices]
        except Exception as e:
            print(f"Error in _get_similar_items: {str(e)}")
            return []

    def _create_outfit(self, filtered_items: List[Dict], budget: float) -> Dict[str, Any]:
        """Create an outfit from filtered items"""
        try:
            outfit_items = []
            total_price = 0
            required_categories = ['tops', 'bottoms', 'shoes']  # Minimum required categories

            # Group items by category
            items_by_category = {}
            for item in filtered_items:
                category = item.get('category', '').lower()
                if category not in items_by_category:
                    items_by_category[category] = []
                items_by_category[category].append(item)

            # Select items for each required category
            for category in required_categories:
                if category in items_by_category and items_by_category[category]:
                    # Sort items by relevance score if available
                    category_items = sorted(
                        items_by_category[category],
                        key=lambda x: x.get('relevance_score', 0),
                        reverse=True
                    )
                    # Select from top items
                    selected_item = random.choice(category_items[:5] if len(category_items) > 5 else category_items)
                    
                    price = float(selected_item.get('price', 0))
                    if total_price + price <= budget:
                        outfit_items.append({
                            'name': selected_item.get('name', ''),
                            'category': category,
                            'price': price,
                            'color': selected_item.get('color', ''),
                            'description': selected_item.get('description', ''),
                            'purchase_link': selected_item.get('purchase_link', ''),
                            'image_data': selected_item.get('image_data')
                        })
                        total_price += price

            # Add accessories if budget allows
            if total_price < budget * 0.8 and 'accessories' in items_by_category:
                accessories = items_by_category['accessories']
                if accessories:
                    accessory = random.choice(accessories)
                    price = float(accessory.get('price', 0))
                    if total_price + price <= budget:
                        outfit_items.append({
                            'name': accessory.get('name', ''),
                            'category': 'accessories',
                            'price': price,
                            'color': accessory.get('color', ''),
                            'description': accessory.get('description', ''),
                            'purchase_link': accessory.get('purchase_link', ''),
                            'image_data': accessory.get('image_data')
                        })
                        total_price += price

            if outfit_items:
                return {
                    'set_id': f'outfit_{random.randint(1000, 9999)}',
                    'total_price': total_price,
                    'items': outfit_items
                }
            return None
        except Exception as e:
            print(f"Error in _create_outfit: {str(e)}")
            return None

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
            # Always use sample data recommendations for now
            return self._get_recommendations_from_samples(
                occasion, budget, preferences, num_recommendations
            )
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
            return self._get_fallback_recommendations(budget, num_recommendations)
    
    def _get_recommendations_from_samples(
        self,
        occasion: str,
        budget: float,
        preferences: Dict[str, Any],
        num_recommendations: int
    ) -> List[Dict[str, Any]]:
        """Generate recommendations using sample data"""
        recommendations = []
        occasion_styles = self.occasion_styles.get(occasion, ['Casual'])
        
        # Convert sample items to list format
        all_items = []
        for category_items in self.sample_items.values():
            all_items.extend(category_items)
        
        # Filter items by occasion and preferences
        filtered_items = []
        for item in all_items:
            if (
                item.price <= budget * 0.4 and
                self._matches_preferences(vars(item), preferences) and
                any(style.lower() in [s.lower() for s in occasion_styles] for style in item.style_tags)
            ):
                filtered_item = vars(item)
                filtered_item['relevance_score'] = 1.0  # Default score
                filtered_items.append(filtered_item)
        
        # Create outfits
        attempts = 0
        max_attempts = num_recommendations * 2
        while len(recommendations) < num_recommendations and attempts < max_attempts:
            outfit = self._create_outfit(filtered_items, budget)
            if outfit and outfit not in recommendations:
                recommendations.append(outfit)
            attempts += 1
        
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
        """Check if an item matches user preferences with fuzzy matching"""
        if not preferences:
            return True
            
        color_prefs = [c.lower() for c in preferences.get('color_preference', [])]
        style_prefs = [s.lower() for s in preferences.get('style_preference', [])]
        
        # Color matching with some flexibility
        item_color = item.get('color', '').lower()
        color_match = not color_prefs or any(
            color in item_color or item_color in color
            for color in color_prefs
        )
        
        # Style matching with some flexibility
        item_styles = [s.lower() for s in item.get('style_tags', [])]
        style_match = not style_prefs or any(
            style in s or s in style
            for s in item_styles
            for style in style_prefs
        )
        
        return color_match and style_match
    
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

if __name__ == "__main__":
    print("Testing WardrobeRecommender with sample data...")
    
    # Initialize recommender
    recommender = WardrobeRecommender()
    
    # Test cases with different scenarios
    test_cases = [
        {
            "occasion": "Business Meeting",
            "budget": 500.0,
            "preferences": {
                "color_preference": ["Black", "Navy", "White"],
                "style_preference": ["Professional", "Classic"]
            }
        },
        {
            "occasion": "Party",
            "budget": 400.0,
            "preferences": {
                "color_preference": ["Gold", "Black"],
                "style_preference": ["Bold", "Trendy"]
            }
        }
    ]
    
    # Run test cases
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['occasion']}")
        print("=" * 50)
        
        recommendations = recommender.get_outfit_recommendations(
            style_profile=torch.zeros(512),  # Dummy style profile
            occasion=test_case["occasion"],
            budget=test_case["budget"],
            preferences=test_case["preferences"]
        )
        
        # Print recommendations
        for j, outfit in enumerate(recommendations, 1):
            print(f"\nOutfit {j} (Total: ${outfit['total_price']:.2f})")
            print("-" * 40)
            for item in outfit["items"]:
                print(f"- {item['name']}")
                print(f"  Price: ${item['price']:.2f}")
                print(f"  Color: {item['color']}")
                print(f"  Description: {item['description']}")
                print() 