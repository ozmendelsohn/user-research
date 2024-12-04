from typing import List, Dict, Any
from llm_client import LLMClient
from persona import Persona
import random

class PersonaGenerator:
    """
    Generates diverse user personas using LLM with function calling.
    """
    
    def __init__(self, provider: str = "groq"):
        """Initialize the PersonaGenerator with specified LLM provider."""
        self.llm = LLMClient(provider=provider)
    
    def _call_with_function(self, prompt: str, function_schema: Dict) -> Dict:
        """Make an LLM call with function calling."""
        return self.llm.function_call(prompt, function_schema)

    def identify_audience_dimensions(self, audience_description: str) -> Dict[str, Any]:
        """
        Identify key dimensions of variation in the target audience.
        """
        function_schema = {
            "name": "analyze_audience_dimensions",
            "description": "Identify key dimensions that differentiate the target audience",
            "parameters": {
                "type": "object",
                "properties": {
                    "dimensions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "values": {
                                    "type": "array",
                                    "items": {"type": "string"}
                                },
                                "importance": {"type": "integer", "minimum": 1, "maximum": 5}
                            },
                            "required": ["name", "description", "values", "importance"]
                        }
                    }
                },
                "required": ["dimensions"]
            }
        }
        
        prompt = f"""Analyze this target audience description and identify 3-5 key dimensions that differentiate different segments:

        Audience: {audience_description}
        
        For each dimension:
        1. Provide a clear name (e.g., "Music Genre Preference", "Usage Context")
        2. Brief description of why this dimension matters
        3. List of distinct values along this dimension
        4. Importance rating (1-5) for persona generation
        
        Think about factors like:
        - Usage patterns and contexts
        - Lifestyle and habits
        - Technical preferences
        - Specific needs and pain points
        - Cultural or demographic factors
        
        Focus on dimensions that would significantly impact product preferences and needs."""
        
        return self._call_with_function(prompt, function_schema)

    def generate_persona(self, 
                        dimension_values: Dict[str, str],
                        audience_context: str) -> Persona:
        """Generate a single persona based on dimension values."""
        function_schema = {
            "name": "create_persona",
            "description": "Create a detailed user persona",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer", "minimum": 18, "maximum": 80},
                    "occupation": {"type": "string"},
                    "interests": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 3,
                        "maxItems": 5
                    },
                    "pain_points": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 4
                    },
                    "tech_savviness": {"type": "integer", "minimum": 1, "maximum": 5},
                    "background": {"type": "string"}
                },
                "required": ["name", "age", "occupation", "interests", "pain_points", 
                           "tech_savviness", "background"]
            }
        }
        
        dimensions_str = "\n".join([f"- {k}: {v}" for k, v in dimension_values.items()])
        prompt = f"""Create a detailed user persona with these characteristics:

        Audience Context: {audience_context}
        
        Key Dimensions:
        {dimensions_str}

        Create a realistic persona that:
        1. Strongly aligns with the specified dimension values
        2. Has a coherent and believable background story that incorporates the dimension values
        3. Includes interests and pain points that naturally relate to the dimension values
        4. Makes the dimension values an integral part of their story
        
        Important: Only use the exact fields specified in the schema. Do not add additional fields.
        The background story should incorporate all dimension-specific details."""
        
        try:
            persona_data = self._call_with_function(prompt, function_schema)
            # Ensure the background includes dimension values
            background = f"{persona_data['background']}\n\nKey Characteristics:\n"
            for dim_name, dim_value in dimension_values.items():
                background += f"- {dim_name}: {dim_value}\n"
            persona_data['background'] = background
            
            return Persona(**persona_data)
        except Exception as e:
            print(f"Error in persona generation: {str(e)}")
            raise

    def generate_diverse_personas(self, 
                                count: int,
                                audience_description: str,
                                ensure_diversity: bool = True,
                                distribution_settings: Dict = None) -> List[Persona]:
        """Generate multiple diverse personas based on audience dimensions."""
        try:
            # First, identify key dimensions of the audience
            dimensions = self.identify_audience_dimensions(audience_description)["dimensions"]
            
            # Sort dimensions by importance
            dimensions.sort(key=lambda x: x["importance"], reverse=True)
            
            personas = []
            dimension_combinations = []
            
            # Generate dimension value combinations
            for i in range(count):
                combination = {}
                for dim in dimensions:
                    if ensure_diversity:
                        # Cycle through values to ensure coverage
                        value_index = i % len(dim["values"])
                        value = dim["values"][value_index]
                    else:
                        # Use distribution settings or random selection
                        if distribution_settings:
                            weights = [
                                distribution_settings.get(f"{dim['name']}_{value}", 1.0/len(dim['values']))
                                for value in dim['values']
                            ]
                            value = random.choices(dim['values'], weights=weights)[0]
                        else:
                            value = random.choice(dim['values'])
                    combination[dim["name"]] = value
                dimension_combinations.append(combination)
            
            # Generate personas based on dimension combinations
            for combination in dimension_combinations:
                persona = self.generate_persona(
                    dimension_values=combination,
                    audience_context=audience_description
                )
                personas.append(persona)
            
            return personas
            
        except Exception as e:
            print(f"Error in dimension-based generation: {str(e)}")
            return self._fallback_generation(count)
    
    def _fallback_generation(self, count: int) -> List[Persona]:
        """Fallback method for basic persona generation."""
        function_schema = {
            "name": "create_basic_persona",
            "description": "Create a basic user persona",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer", "minimum": 18, "maximum": 80},
                    "occupation": {"type": "string"},
                    "interests": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 3,
                        "maxItems": 5
                    },
                    "pain_points": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 4
                    },
                    "tech_savviness": {"type": "integer", "minimum": 1, "maximum": 5},
                    "background": {"type": "string"}
                },
                "required": ["name", "age", "occupation", "interests", "pain_points", 
                           "tech_savviness", "background"]
            }
        }
        
        personas = []
        for i in range(count):
            prompt = f"""Create a realistic user persona with diverse characteristics.
            Make the persona feel like a real person with:
            1. A realistic full name
            2. Age between 18-80
            3. Specific occupation
            4. 3-5 relevant interests/hobbies
            5. 2-4 specific pain points
            6. Tech savviness rating (1-5)
            7. Brief but detailed background story
            
            Make this persona ({i+1} of {count}) distinct from others."""
            
            try:
                persona_data = self._call_with_function(prompt, function_schema)
                personas.append(Persona(**persona_data))
            except Exception as e:
                print(f"Error generating fallback persona {i+1}: {str(e)}")
                continue
        
        return personas