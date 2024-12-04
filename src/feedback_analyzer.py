from typing import Dict, List, Any
from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from config import LLM_MODEL

class FeedbackAnalyzer:
    """
    Analyzes and aggregates feedback from multiple personas.
    """
    
    def __init__(self):
        """Initialize the FeedbackAnalyzer."""
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def _call_with_function(self, prompt: str, function_schema: Dict) -> Dict:
        """Make an OpenAI API call with function calling."""
        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            functions=[function_schema],
            function_call={"name": function_schema["name"]}
        )
        return json.loads(response.choices[0].message.function_call.arguments)
    
    def analyze_sentiment(self, feedback_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyze sentiment and key points from each persona's feedback.
        """
        function_schema = {
            "name": "analyze_feedback_sentiment",
            "description": "Analyze sentiment and key points from feedback",
            "parameters": {
                "type": "object",
                "properties": {
                    "sentiment_score": {"type": "number", "minimum": -1, "maximum": 1},
                    "positive_points": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "concerns": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "suggested_improvements": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["sentiment_score", "positive_points", "concerns", 
                           "suggested_improvements"]
            }
        }
        
        results = {}
        for persona_name, feedback in feedback_data.items():
            prompt = f"""Analyze this feedback from {persona_name}:
            
            {feedback}
            
            Provide:
            1. A sentiment score (-1 to 1)
            2. Key positive points mentioned
            3. Main concerns raised
            4. Suggested improvements
            
            Be specific and extract actual points from the feedback."""
            
            results[persona_name] = self._call_with_function(prompt, function_schema)
        
        return results
    
    def aggregate_insights(self, 
                         product_description: str,
                         feedback_data: Dict[str, str],
                         sentiment_analysis: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Aggregate insights from all feedback and generate recommendations.
        """
        function_schema = {
            "name": "generate_insights",
            "description": "Generate aggregated insights from feedback",
            "parameters": {
                "type": "object",
                "properties": {
                    "overall_sentiment": {
                        "type": "object",
                        "properties": {
                            "score": {"type": "number", "minimum": -1, "maximum": 1},
                            "summary": {"type": "string"}
                        }
                    },
                    "key_strengths": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "common_concerns": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "market_segments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "segment": {"type": "string"},
                                "receptiveness": {"type": "string"},
                                "key_needs": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    },
                    "recommendations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "area": {"type": "string"},
                                "suggestion": {"type": "string"},
                                "priority": {"type": "string", 
                                           "enum": ["High", "Medium", "Low"]}
                            }
                        }
                    }
                },
                "required": ["overall_sentiment", "key_strengths", "common_concerns",
                           "market_segments", "recommendations"]
            }
        }
        
        # Prepare the analysis prompt
        sentiment_summary = "\n".join([
            f"{name}:\n- Score: {data['sentiment_score']}\n- Positives: {', '.join(data['positive_points'])}\n- Concerns: {', '.join(data['concerns'])}"
            for name, data in sentiment_analysis.items()
        ])
        
        prompt = f"""Analyze this product feedback and generate comprehensive insights:

        Product Description:
        {product_description}

        Sentiment Analysis Summary:
        {sentiment_summary}

        Raw Feedback:
        {json.dumps(feedback_data, indent=2)}

        Generate:
        1. Overall sentiment and reception
        2. Key product strengths
        3. Common concerns across personas
        4. Market segment analysis
        5. Prioritized recommendations for improvement
        
        Focus on actionable insights and clear patterns in the feedback."""
        
        return self._call_with_function(prompt, function_schema)
    
    def generate_report(self, 
                       product_description: str,
                       feedback_data: Dict[str, str]) -> Dict[str, Any]:
        """
        Generate a comprehensive analysis report.
        """
        # Step 1: Analyze sentiment for each persona
        sentiment_analysis = self.analyze_sentiment(feedback_data)
        
        # Step 2: Aggregate insights
        insights = self.aggregate_insights(
            product_description,
            feedback_data,
            sentiment_analysis
        )
        
        # Step 3: Compile full report
        return {
            "product_description": product_description,
            "individual_analysis": sentiment_analysis,
            "aggregated_insights": insights
        } 