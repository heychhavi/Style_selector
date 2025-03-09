from datasets import load_dataset
import json

def analyze_dataset():
    # Load the dataset
    print("Loading dataset...")
    ds = load_dataset("Marqo/polyvore")
    
    # Print basic information
    print("\nDataset splits:", ds.keys())
    print("\nTraining set size:", len(ds['train']))
    
    # Get features information
    print("\nFeatures:", ds['train'].features)
    
    # Print first example
    print("\nFirst example:")
    first_item = ds['train'][0]
    print(json.dumps(first_item, indent=2))
    
    # Analyze unique categories
    categories = set()
    for item in ds['train']:
        if 'category' in item:
            categories.add(item['category'])
    
    print("\nUnique categories:", sorted(list(categories)))

if __name__ == "__main__":
    analyze_dataset() 