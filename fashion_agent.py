import openai
from typing import Dict, Any, List, Optional
import json
import os

class FashionAgent:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.conversation_history = []
        self.user_context = {}
        self.style_profile = {
            "style_preferences": [],
            "color_preferences": [],
            "typical_outfits": [],
            "comfort_priorities": [],
            "budget_range": None,
            "occasions": []
        }
        self.key_questions = [
            "What's your typical budget range for clothing items? (e.g., tops, bottoms, dresses)",
            "Which occasions do you most often dress for? (e.g., work, casual, formal events)",
            "What colors do you feel most confident wearing?",
            "Are there any specific style elements you want to incorporate or avoid?"
        ]
        self.current_question_index = 0
        
    def _format_user_data(self, 
                         style_profile: Optional[Any], 
                         occasion: str, 
                         preferences: Dict[str, Any], 
                         uploaded_images: List[str]) -> str:
        """Format user data for the agent"""
        context = {
            "occasion": occasion,
            "preferences": {
                "colors": preferences.get("color_preference", []),
                "styles": preferences.get("style_preference", []),
                "budget": preferences.get("budget", 0)
            },
            "has_style_profile": style_profile is not None,
            "uploaded_images_count": len(uploaded_images) if uploaded_images else 0,
            "style_profile_summary": self._summarize_style_profile(style_profile) if style_profile is not None else None
        }
        self.user_context = context
        return json.dumps(context, indent=2)
    
    def _summarize_style_profile(self, style_profile: Any) -> Dict[str, Any]:
        """Summarize the style profile features"""
        if style_profile is None:
            return None
        
        try:
            # Convert tensor to list if needed
            profile_data = style_profile.tolist() if hasattr(style_profile, 'tolist') else style_profile
            return {
                "feature_vector": profile_data,
                "vector_size": len(profile_data)
            }
        except Exception as e:
            print(f"Error summarizing style profile: {str(e)}")
            return None
        
    def _process_agent_response(self, response: str) -> Dict[str, Any]:
        """Process the agent's response into structured recommendations"""
        try:
            if not response or not response.strip():
                return {
                    "style_description": "",
                    "style_tags": [],
                    "color_suggestions": [],
                    "occasion_specific_tips": [],
                    "budget_allocation": {},
                    "personalized_advice": "No recommendations available at the moment.",
                    "seasonal_recommendations": [],
                    "style_combinations": [],
                    "conversation_response": "I apologize, but I couldn't generate recommendations at this time."
                }
                
            # First try to parse as JSON
            try:
                enhanced_preferences = json.loads(response)
            except json.JSONDecodeError:
                # If not JSON, treat as conversation response
                return {
                    "style_description": "",
                    "style_tags": [],
                    "color_suggestions": [],
                    "occasion_specific_tips": [],
                    "budget_allocation": {},
                    "personalized_advice": response,
                    "seasonal_recommendations": [],
                    "style_combinations": [],
                    "conversation_response": response
                }

            return {
                "style_tags": enhanced_preferences.get("style_tags", []),
                "color_suggestions": enhanced_preferences.get("color_suggestions", []),
                "occasion_specific_tips": enhanced_preferences.get("occasion_specific_tips", []),
                "budget_allocation": enhanced_preferences.get("budget_allocation", {}),
                "style_description": enhanced_preferences.get("style_description", ""),
                "personalized_advice": enhanced_preferences.get("personalized_advice", ""),
                "seasonal_recommendations": enhanced_preferences.get("seasonal_recommendations", []),
                "style_combinations": enhanced_preferences.get("style_combinations", []),
                "conversation_response": enhanced_preferences.get("conversation_response", "")
            }
            
        except Exception as e:
            print(f"Error processing response: {str(e)}")
            return {
                "style_description": "",
                "style_tags": [],
                "color_suggestions": [],
                "occasion_specific_tips": [],
                "budget_allocation": {},
                "personalized_advice": "An error occurred while processing recommendations.",
                "seasonal_recommendations": [],
                "style_combinations": [],
                "conversation_response": "I apologize, but there was an error processing the response."
            }

    async def chat(self, message: str) -> str:
        """Have a conversation with the fashion agent"""
        try:
            # Create context-aware prompt
            context_prompt = ""
            if self.user_context:
                context_prompt = f"\nUser Context:\n{json.dumps(self.user_context, indent=2)}"
            
            # Add message to conversation history
            self.conversation_history.append({"role": "user", "content": message})
            
            # Prepare messages for OpenAI
            messages = [
                {"role": "system", "content": """You are a friendly and knowledgeable fashion stylist assistant. 
                Your goal is to provide helpful, specific fashion advice while maintaining a conversational tone. 
                Consider the user's preferences and context when available."""},
                *[{"role": msg["role"], "content": msg["content"]} for msg in self.conversation_history[-10:]]  # Last 10 messages
            ]
            
            # Get completion from OpenAI
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            # Extract response
            assistant_response = response.choices[0].message.content
            
            # Add response to conversation history
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            return assistant_response
            
        except Exception as e:
            print(f"Error in chat: {str(e)}")
            return "I apologize, but I'm having trouble processing your request at the moment. Could you please try again?"

    async def analyze_style_from_images(self, image_analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user's style based on uploaded images"""
        try:
            # Create prompt for image analysis
            prompt = f"""As a fashion stylist, analyze these image features and provide style insights:
            Image Analysis Data: {json.dumps(image_analysis_result, indent=2)}
            
            Please provide a structured analysis including:
            1. Dominant style elements observed
            2. Color palette preferences
            3. Fit and silhouette preferences
            4. Level of formality
            5. Notable patterns or textures
            
            Return the analysis in JSON format."""

            messages = [
                {"role": "system", "content": "You are an expert fashion stylist analyzing user's style from images."},
                {"role": "user", "content": prompt}
            ]
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            analysis = json.loads(response.choices[0].message.content)
            self.style_profile.update(analysis)
            return analysis
            
        except Exception as e:
            print(f"Error analyzing images: {str(e)}")
            return {}

    async def get_next_question(self) -> Dict[str, Any]:
        """Get the next question in the style interview"""
        try:
            if self.current_question_index >= len(self.key_questions):
                return {
                    "is_complete": True,
                    "next_question": None,
                    "status": "complete"
                }
                
            return {
                "is_complete": False,
                "next_question": self.key_questions[self.current_question_index],
                "status": "in_progress"
            }
        except Exception as e:
            print(f"Error getting next question: {str(e)}")
            return {
                "is_complete": False,
                "next_question": "What's your typical budget range for clothing items?",  # Default first question
                "status": "error",
                "error": str(e)
            }

    async def process_response(self, user_response: str, image_analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process user's response and update style profile"""
        try:
            # Add response to conversation history
            current_question = self.key_questions[self.current_question_index]
            print(f"Processing response for question: {current_question}")
            print(f"User response: {user_response}")
            
            # Create prompt to extract information
            prompt = f"""Based on the user's response to "{current_question}":
            User Response: "{user_response}"
            
            Extract the relevant information and provide a response in the following JSON format:
            {{
                "preferences": {{
                    "style_preferences": ["list", "of", "style", "elements"],
                    "color_preferences": ["list", "of", "colors"],
                    "budget_range": "budget information",
                    "occasions": ["list", "of", "occasions"],
                    "comfort_priorities": ["list", "of", "priorities"]
                }},
                "analysis": "brief analysis of the response"
            }}

            Only include fields that are relevant to the current question. For example, if asking about budget, focus on budget_range."""

            messages = [
                {"role": "system", "content": "You are a fashion stylist extracting style preferences from user responses. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ]
            
            print("Sending request to OpenAI...")
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=messages,
                temperature=0.3,
                max_tokens=500
            )
            
            # Extract and clean response content
            response_content = response.choices[0].message.content.strip()
            print(f"Raw OpenAI response: {response_content}")
            
            try:
                # Try to find JSON content if it's wrapped in other text
                json_start = response_content.find('{')
                json_end = response_content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    response_content = response_content[json_start:json_end]
                    print(f"Extracted JSON content: {response_content}")
                
                # Parse the JSON response
                extracted_info = json.loads(response_content)
                print(f"Parsed JSON: {json.dumps(extracted_info, indent=2)}")
                
                # Update style profile with extracted information
                if isinstance(extracted_info, dict) and 'preferences' in extracted_info:
                    preferences = extracted_info['preferences']
                    for key, value in preferences.items():
                        if value and value != []:  # Only update if value is not empty
                            print(f"Updating {key} with {value}")
                            self.style_profile[key] = value
                    print(f"Updated style profile: {json.dumps(self.style_profile, indent=2)}")
                else:
                    print(f"Warning: Unexpected response format: {extracted_info}")
                    return {
                        "is_complete": False,
                        "error": "Unexpected response format from AI",
                        "next_question": current_question,
                        "current_profile": self.style_profile
                    }
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {str(e)}")
                print(f"Problematic content: {response_content}")
                return {
                    "is_complete": False,
                    "error": "Could not process the response. Please try again.",
                    "next_question": current_question,
                    "current_profile": self.style_profile
                }
            
            # Increment question index
            self.current_question_index += 1
            print(f"Moving to next question index: {self.current_question_index}")
            
            # If this was the last question, generate final profile
            if self.current_question_index >= len(self.key_questions):
                print("Completed all questions")
                if image_analysis:
                    self.style_profile.update(image_analysis)
                    print("Added image analysis to profile")
                
                return {
                    "is_complete": True,
                    "style_profile": self.style_profile,
                    "next_question": None,
                    "message": "Style profile complete!"
                }
            
            # Return next question
            next_question = self.key_questions[self.current_question_index]
            print(f"Next question: {next_question}")
            return {
                "is_complete": False,
                "next_question": next_question,
                "current_profile": self.style_profile,
                "message": "Thanks! Let's continue with the next question."
            }
            
        except Exception as e:
            print(f"Error in process_response: {str(e)}")
            return {
                "is_complete": False,
                "error": str(e),
                "next_question": self.key_questions[min(self.current_question_index, len(self.key_questions)-1)],
                "current_profile": self.style_profile
            }

    async def get_recommendations(self, occasion: str) -> Dict[str, Any]:
        """Get personalized recommendations based on style profile and occasion"""
        try:
            # Create prompt for recommendations
            prompt = f"""As a fashion stylist, provide outfit recommendations based on:
            Style Profile: {json.dumps(self.style_profile, indent=2)}
            Occasion: {occasion}
            
            Provide recommendations in JSON format including:
            1. Complete outfits with specific items
            2. Color combinations
            3. Style elements to incorporate
            4. Styling tips
            5. Budget allocation
            
            Ensure recommendations match the user's:
            - Stated budget range
            - Color preferences
            - Style preferences
            - Comfort priorities
            - Typical outfit choices"""

            messages = [
                {"role": "system", "content": "You are an expert fashion stylist providing personalized recommendations."},
                {"role": "user", "content": prompt}
            ]
            
            response = await openai.ChatCompletion.acreate(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            recommendations = json.loads(response.choices[0].message.content)
            return recommendations
            
        except Exception as e:
            print(f"Error getting recommendations: {str(e)}")
            return {
                "error": "Unable to generate recommendations at this time."
            } 