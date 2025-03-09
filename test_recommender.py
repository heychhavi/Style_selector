from wardrobe_recommender import WardrobeRecommender
import torch

def test_business_outfit():
    recommender = WardrobeRecommender()
    preferences = {
        "color_preference": ["Black", "Navy", "White"],
        "style_preference": ["Professional", "Classic", "Formal"]
    }
    recommendations = recommender.get_outfit_recommendations(
        style_profile=torch.zeros(512),
        occasion="Business Meeting",
        budget=500.0,
        preferences=preferences
    )
    print("\nBusiness Meeting Outfit Recommendations:")
    print_recommendations(recommendations)

def test_casual_outfit():
    recommender = WardrobeRecommender()
    preferences = {
        "color_preference": ["Blue", "Gray", "White"],
        "style_preference": ["Casual", "Comfortable", "Streetwear"]
    }
    recommendations = recommender.get_outfit_recommendations(
        style_profile=torch.zeros(512),
        occasion="Casual Outing",
        budget=300.0,
        preferences=preferences
    )
    print("\nCasual Outing Outfit Recommendations:")
    print_recommendations(recommendations)

def test_party_outfit():
    recommender = WardrobeRecommender()
    preferences = {
        "color_preference": ["Black", "Red", "Gold"],
        "style_preference": ["Trendy", "Stylish", "Bold"]
    }
    recommendations = recommender.get_outfit_recommendations(
        style_profile=torch.zeros(512),
        occasion="Party",
        budget=400.0,
        preferences=preferences
    )
    print("\nParty Outfit Recommendations:")
    print_recommendations(recommendations)

def print_recommendations(recommendations):
    print("===================")
    for i, outfit in enumerate(recommendations, 1):
        print(f"\nOutfit {i} (Total: ${outfit['total_price']:.2f}):")
        print("-" * 40)
        for item in outfit['items']:
            print(f"- {item['name']}")
            print(f"  Category: {item['category']}")
            print(f"  Price: ${item['price']:.2f}")
            print(f"  Color: {item['color']}")
            print(f"  Description: {item['description']}")
            print(f"  Purchase Link: {item['purchase_link']}")
            print()

if __name__ == "__main__":
    print("Testing outfit recommendations for different occasions...")
    test_business_outfit()
    test_casual_outfit()
    test_party_outfit() 