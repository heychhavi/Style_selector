import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import numpy as np
from io import BytesIO
import base64

class OutfitVisualizer:
    def __init__(self):
        # Initialize Stable Diffusion pipeline for outfit visualization
        try:
            self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                "runwayml/stable-diffusion-inpainting",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            if torch.cuda.is_available():
                self.inpaint_pipeline.to("cuda")
            print("Outfit visualizer initialized successfully")
        except Exception as e:
            print(f"Error initializing outfit visualizer: {e}")
            self.inpaint_pipeline = None

    def generate_outfit_preview(
        self,
        user_image: Image.Image,
        outfit_items: list,
        prompt_prefix: str = "high quality fashion photo of person wearing"
    ) -> str:
        """
        Generate a preview of how the outfit would look on the user
        
        Args:
            user_image: PIL Image of the user
            outfit_items: List of outfit items with descriptions
            prompt_prefix: Prefix for the generation prompt
            
        Returns:
            Base64 encoded string of the generated image
        """
        try:
            if self.inpaint_pipeline is None:
                return None

            # Prepare the user image
            user_image = user_image.convert("RGB")
            user_image = user_image.resize((512, 512))

            # Create outfit description
            outfit_description = ", ".join([
                f"{item['color']} {item['name'].lower()}"
                for item in outfit_items
            ])

            # Create the prompt
            prompt = f"{prompt_prefix} {outfit_description}, professional fashion photography, high quality, detailed clothing"
            negative_prompt = "low quality, blurry, distorted, unrealistic, bad proportions"

            # Generate the visualization
            output_image = self.inpaint_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=user_image,
                num_inference_steps=30,
                guidance_scale=7.5
            ).images[0]

            # Convert to base64
            buffered = BytesIO()
            output_image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            return img_str

        except Exception as e:
            print(f"Error generating outfit preview: {e}")
            return None

    def create_outfit_collage(
        self,
        user_image: Image.Image,
        outfit_items: list,
        generated_preview: Image.Image = None
    ) -> str:
        """
        Create a collage showing the user's image, individual items, and the generated preview
        
        Args:
            user_image: PIL Image of the user
            outfit_items: List of outfit items with their images
            generated_preview: Generated preview image (optional)
            
        Returns:
            Base64 encoded string of the collage image
        """
        try:
            # Create a white background
            collage = Image.new('RGB', (1024, 512), 'white')
            
            # Resize user image
            user_image = user_image.resize((256, 256))
            collage.paste(user_image, (0, 128))
            
            # Add outfit items
            x_offset = 256
            for item in outfit_items:
                if item.get('image_data'):
                    try:
                        item_img = Image.open(BytesIO(base64.b64decode(item['image_data'])))
                        item_img = item_img.resize((256, 256))
                        collage.paste(item_img, (x_offset, 128))
                        x_offset += 256
                    except Exception as e:
                        print(f"Error adding item to collage: {e}")
                        continue
            
            # Add generated preview if available
            if generated_preview:
                preview_img = generated_preview.resize((256, 256))
                collage.paste(preview_img, (768, 128))
            
            # Convert to base64
            buffered = BytesIO()
            collage.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return img_str
            
        except Exception as e:
            print(f"Error creating outfit collage: {e}")
            return None 