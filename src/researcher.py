from typing import List, Dict
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from PIL import Image
import base64
import io

from persona import Persona
from config import LLM_MODEL, MAX_TOKENS

class ProductResearcher:
    """
    A class to conduct user research using LLM agents with different personas.
    """
    
    def __init__(self):
        """
        Initialize the ProductResearcher with OpenAI credentials.
        """
        load_dotenv()
        self.llm = ChatOpenAI(
            model=LLM_MODEL,
            max_tokens=MAX_TOKENS
        )
        
    def _encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 string.
        
        Parameters
        ----------
        image_path : str
            Path to the image file
            
        Returns
        -------
        str
            Base64 encoded image string
        """
        with Image.open(image_path) as img:
            buffer = io.BytesIO()
            img.save(buffer, format=img.format)
            return base64.b64encode(buffer.getvalue()).decode()

    def analyze_product(self, 
                       personas: List[Persona], 
                       product_description: str,
                       product_images: List[str] = None) -> Dict[str, str]:
        """
        Analyze a product from multiple persona perspectives.
        """
        results = {}
        
        for persona in personas:
            # Create base message content
            message_content = [
                {
                    "type": "text",
                    "text": f"""You are {persona.name}, {persona.age} years old, working as {persona.occupation}. 
                    Consider the following aspects about yourself:
                    - Interests: {', '.join(persona.interests)}
                    - Pain points: {', '.join(persona.pain_points)}
                    - Technical proficiency: {persona.tech_savviness}/5
                    - Background: {persona.background}
                    
                    Analyze the following product from your perspective. Consider:
                    1. Would you use this product? Why or why not?
                    2. What features appeal to you most?
                    3. What concerns do you have?
                    4. How much would you be willing to pay?
                    5. What improvements would make this more appealing to you?
                    
                    Product description: {product_description}
                    """
                }
            ]
            
            # Add images if provided
            if product_images:
                for image_path in product_images:
                    encoded_image = self._encode_image(image_path)
                    message_content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_image}"
                        }
                    })
            
            messages = [
                {
                    "role": "user",
                    "content": message_content
                }
            ]
            
            response = self.llm.invoke(messages)
            results[persona.name] = response.content
            
        return results 