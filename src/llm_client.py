from typing import Dict, List, Optional, Union, Type, Any
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.output_parsers import OutputFixingParser
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, create_model, Field
import json
import os
import logging
import traceback
from dotenv import load_dotenv
from config import LLM_SETTINGS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMError(Exception):
    """Base exception class for LLM-related errors."""
    pass

class DimensionModel(BaseModel):
    """Model for a single dimension."""
    name: str = Field(description="Name of the dimension")
    description: str = Field(description="Description of why this dimension matters")
    values: List[str] = Field(description="Possible values for this dimension")
    importance: int = Field(description="Importance rating from 1 to 5", ge=1, le=5)

class DimensionsResponse(BaseModel):
    """Model for the dimensions response."""
    dimensions: List[DimensionModel] = Field(description="List of dimensions")

class LLMClient:
    """
    Unified client for LLM interactions using LangChain with structured output.
    """
    
    def __init__(self, provider: str = "groq"):
        """Initialize the LLM client with specified provider."""
        load_dotenv()
        self.provider = provider
        
        try:
            settings = LLM_SETTINGS[provider]
            
            if provider == "groq":
                self.llm = ChatGroq(
                    groq_api_key=os.getenv("GROQ_API_KEY"),
                    model=settings["model"],
                    temperature=settings["temperature"],
                    streaming=True
                )
            elif provider == "ollama":
                self.llm = ChatOllama(
                    model=settings["model"],
                    temperature=settings["temperature"],
                    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
                )
            else:
                self.llm = ChatOpenAI(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model=settings["model"],
                    temperature=settings["temperature"]
                )
            logger.info(f"Initialized LLM client with provider: {provider}")
            
        except Exception as e:
            error_msg = f"Failed to initialize LLM client: {str(e)}"
            logger.error(error_msg)
            logger.debug(f"Traceback: {traceback.format_exc()}")
            raise LLMError(error_msg)
    
    def _create_output_parser(self, schema: Dict, provider: str):
        """Create appropriate output parser based on provider."""
        json_parser = JsonOutputParser(pydantic_object=schema)
        
        if provider == "ollama":
            # Use OutputFixingParser for Ollama to handle potential JSON issues
            return OutputFixingParser.from_llm(
                parser=json_parser,
                llm=self.llm
            )
        return json_parser
    
    def _fix_incomplete_schema(self, result: Dict, max_attempts: int = 2) -> Dict:
        """
        Attempt to fix incomplete schema by reprompting the model.
        
        Parameters
        ----------
        result : Dict
            The incomplete result
        max_attempts : int
            Maximum number of reprompting attempts
        """
        logger.info("Attempting to fix incomplete schema")
        
        # Create a fixing prompt based on what's missing
        def create_fixing_prompt(original_result, missing_fields):
            return f"""Fix this incomplete JSON by adding the missing fields: {', '.join(missing_fields)}
            
            Current JSON:
            {json.dumps(original_result, indent=2)}
            
            Required fields for each dimension:
            - name (string)
            - description (string)
            - values (array of strings)
            - importance (integer 1-5)
            
            Return the complete JSON with all fields."""
        
        current_result = result
        attempts = 0
        
        while attempts < max_attempts:
            try:
                # Validate current structure
                if not isinstance(current_result, dict):
                    current_result = {"dimensions": current_result if isinstance(current_result, list) else []}
                
                dimensions = current_result.get("dimensions", [])
                if not dimensions:
                    logger.error("No dimensions found")
                    return self._get_fallback_dimensions()
                
                # Check each dimension for missing fields
                required_fields = {"name", "description", "values", "importance"}
                fixed_dimensions = []
                needs_fixing = False
                
                for dim in dimensions:
                    missing_fields = required_fields - set(dim.keys())
                    if missing_fields:
                        needs_fixing = True
                        logger.info(f"Missing fields in dimension: {missing_fields}")
                        
                        # Create fixing prompt for this dimension
                        fixing_prompt = create_fixing_prompt(dim, missing_fields)
                        
                        # Try to fix this dimension
                        fixed_response = self.llm.invoke([
                            {"role": "system", "content": "You are a JSON fixing bot. Return only valid JSON."},
                            {"role": "user", "content": fixing_prompt}
                        ])
                        
                        try:
                            fixed_dim = json.loads(fixed_response.content)
                            if isinstance(fixed_dim, dict) and all(field in fixed_dim for field in required_fields):
                                fixed_dimensions.append(fixed_dim)
                            else:
                                # Create a minimal valid dimension
                                fixed_dimensions.append({
                                    "name": dim.get("name", "Unknown Dimension"),
                                    "description": dim.get("description", "No description available"),
                                    "values": dim.get("values", ["Value 1", "Value 2"]),
                                    "importance": dim.get("importance", 3)
                                })
                        except (json.JSONDecodeError, AttributeError):
                            # If fixing failed, create a minimal valid dimension
                            fixed_dimensions.append({
                                "name": dim.get("name", "Unknown Dimension"),
                                "description": dim.get("description", "No description available"),
                                "values": dim.get("values", ["Value 1", "Value 2"]),
                                "importance": dim.get("importance", 3)
                            })
                    else:
                        fixed_dimensions.append(dim)
                
                if not needs_fixing:
                    return current_result
                
                current_result = {"dimensions": fixed_dimensions}
                
                # Validate the fixed result
                DimensionsResponse(**current_result)
                logger.info("Successfully fixed schema")
                return current_result
                
            except Exception as e:
                logger.warning(f"Fix attempt {attempts + 1} failed: {str(e)}")
                attempts += 1
        
        logger.error("Failed to fix schema after all attempts")
        return self._get_fallback_dimensions()
    
    def function_call(self, 
                     prompt: str, 
                     function_schema: Dict,
                     temperature: Optional[float] = None) -> Dict:
        """Make an LLM call with structured output using LangChain."""
        try:
            logger.info(f"Making function call with {self.provider}")
            
            # For Ollama, we'll use a more explicit prompt structure
            if self.provider == "ollama":
                system_prompt = """You are a JSON-only response bot. You must:
                1. ALWAYS respond with valid JSON
                2. NEVER include any other text
                3. EXACTLY match the required schema
                4. Include ALL required fields
                5. Use the exact field names shown
                
                The response MUST be a JSON object with a 'dimensions' array containing objects with:
                - name (string)
                - description (string)
                - values (array of strings)
                - importance (integer 1-5)"""
                
                # Create a more explicit example
                example = {
                    "dimensions": [
                        {
                            "name": "Usage Context",
                            "description": "Primary situation where the product is used",
                            "values": ["Work", "Home", "Travel"],
                            "importance": 5
                        },
                        {
                            "name": "Tech Level",
                            "description": "User's comfort with technology",
                            "values": ["Basic", "Intermediate", "Advanced"],
                            "importance": 4
                        }
                    ]
                }
                
                formatted_prompt = f"""TASK:
                Analyze this audience and identify key differentiating dimensions:
                {prompt}

                REQUIRED FORMAT:
                {json.dumps(example, indent=2)}

                RULES:
                - Return 3-5 dimensions
                - Each dimension needs all fields shown above
                - Values must be specific and distinct
                - Importance must be 1-5
                - Response must be valid JSON only
                - Do not include any other text

                RESPOND WITH JSON ONLY:"""
                
            else:
                # Original prompt for other providers
                system_prompt = """You are a helpful assistant that provides structured responses.
                Ensure your response matches the required format exactly."""
                
                formatted_prompt = f"""
                Required JSON Schema:
                {json.dumps(function_schema['parameters'], indent=2)}

                Task:
                {prompt}

                Remember: Response must be valid JSON matching the schema exactly.
                """
            
            # Create output parser
            if function_schema["name"] == "analyze_audience_dimensions":
                output_parser = self._create_output_parser(DimensionsResponse, self.provider)
            else:
                model = self._create_pydantic_model(function_schema)
                output_parser = self._create_output_parser(model, self.provider)
            
            # Create chat template
            chat_template = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_prompt),
                HumanMessage(content=formatted_prompt)
            ])
            
            # Create and execute chain
            chain = chat_template | self.llm | output_parser
            
            try:
                logger.info("Generating response...")
                result = chain.invoke({})
                logger.info("Successfully generated response")
                logger.debug(f"Raw response: {result}")
                
                # For Ollama, ensure we have the correct structure
                if self.provider == "ollama":
                    if isinstance(result, str):
                        result = json.loads(result)
                    if not isinstance(result, dict):
                        if isinstance(result, list):
                            result = {"dimensions": result}
                        else:
                            raise LLMError(f"Expected dict or list, got {type(result)}")
                    
                    # Try to fix incomplete schema before falling back
                    try:
                        DimensionsResponse(**result)
                    except Exception as validation_error:
                        logger.warning(f"Schema validation failed: {str(validation_error)}")
                        result = self._fix_incomplete_schema(result)
                
                logger.info("Response validated successfully")
                return result
                
            except Exception as e:
                error_msg = f"Chain execution failed with {self.provider}: {str(e)}"
                logger.error(error_msg)
                logger.debug(f"Traceback: {traceback.format_exc()}")
                
                if function_schema["name"] == "analyze_audience_dimensions":
                    logger.info(f"Using fallback dimensions for {self.provider}")
                    return self._get_fallback_dimensions()
                
                raise LLMError(error_msg)
                
        except Exception as e:
            error_msg = f"Function call failed with {self.provider}: {str(e)}"
            logger.error(error_msg)
            logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # Always return fallback dimensions for audience analysis
            if function_schema["name"] == "analyze_audience_dimensions":
                return self._get_fallback_dimensions()
            raise LLMError(error_msg)
    
    def _get_fallback_dimensions(self) -> Dict:
        """Return fallback dimensions when analysis fails."""
        logger.info("Providing fallback dimensions")
        return {
            "dimensions": [
                {
                    "name": "Work Style",
                    "description": "Preferred approach to work organization",
                    "values": ["Structured", "Flexible", "Hybrid"],
                    "importance": 4
                },
                {
                    "name": "Tech Adoption",
                    "description": "Attitude towards new technology",
                    "values": ["Early Adopter", "Pragmatic", "Conservative"],
                    "importance": 4
                },
                {
                    "name": "Work Environment",
                    "description": "Primary work context",
                    "values": ["Office", "Remote", "Hybrid"],
                    "importance": 3
                },
                {
                    "name": "Team Size",
                    "description": "Typical team collaboration context",
                    "values": ["Solo", "Small Team", "Large Team"],
                    "importance": 3
                }
            ]
        }
    
    def _create_pydantic_model(self, schema: Dict) -> Type[BaseModel]:
        """
        Create a Pydantic model from a JSON schema.
        
        Parameters
        ----------
        schema : Dict
            JSON schema definition
            
        Returns
        -------
        Type[BaseModel]
            Generated Pydantic model
        """
        properties = schema["parameters"]["properties"]
        fields = {}
        
        for field_name, field_schema in properties.items():
            field_type = self._get_field_type(field_schema)
            field_desc = field_schema.get("description", "")
            fields[field_name] = (field_type, Field(description=field_desc))
        
        return create_model(
            schema["name"],
            **fields,
            __doc__=schema.get("description", "Generated model from schema")
        )
    
    def _get_field_type(self, field_schema: Dict) -> Type:
        """Convert JSON schema types to Python types."""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": float,
            "boolean": bool,
            "array": List,
            "object": Dict
        }
        
        field_type = field_schema["type"]
        if field_type == "array":
            item_type = self._get_field_type(field_schema["items"])
            return List[item_type]
        return type_mapping.get(field_type, Any)    
