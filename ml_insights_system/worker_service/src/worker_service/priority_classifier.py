import os
import logging
import openai
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class PriorityClassifier:
    """
    A class that uses OpenAI's API to classify customer sentences 
    as high priority (churn or buying intent) or normal priority.
    """
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        logger.info("PriorityClassifier initialized with OpenAI API")
    
    async def classify_priority(self, text: str) -> dict:
        """
        Classify the priority of a customer sentence.
        
        Args:
            text: The customer sentence to classify
            
        Returns:
            dict: A dictionary containing priority classification and reasoning
        """
        try:
            logger.info(f"Classifying priority for text: {text[:30]}...")
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # You can adjust the model as needed
                messages=[
                    {"role": "system", "content": """
                    You are a customer intent classifier for a business. Your task is to determine if a customer 
                    message indicates either:
                    
                    1. Churn intent - Any indication that the customer might leave, is unhappy, or is considering 
                       competitors. Examples: cancellation queries, complaints, dissatisfaction, or competitor mentions.
                    
                    2. Buying intent - Any indication the customer is interested in purchasing products or services.
                       This includes early-stage buying signals like asking for information, quotations, pricing,
                       product details, or expressing direct interest in purchasing.
                    
                    If EITHER churn intent OR buying intent is detected, classify as "high" priority.
                    If NEITHER type of intent is detected, classify as "normal" priority.
                    
                    Return ONLY a JSON object with the following fields:
                    - priority: "high" or "normal"
                    - intent_type: "churn", "buying", or "none"
                    - reasoning: Brief explanation for the classification
                    """},
                    {"role": "user", "content": text}
                ],
                temperature=0,
                response_format={"type": "json_object"}
            )
            
            # Extract the JSON response
            result = response.choices[0].message.content
            
            # Log the classification result
            logger.info(f"Classification result: {result}")
            
            # Return the parsed JSON
            import json
            return json.loads(result)
            
        except Exception as e:
            logger.error(f"Error in classification: {str(e)}")
            # Return a default result in case of error
            return {
                "priority": "normal",
                "intent_type": "none",
                "reasoning": f"Classification error: {str(e)}"
            }