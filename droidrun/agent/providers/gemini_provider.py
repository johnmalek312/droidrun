"""
Gemini provider implementation.
"""

import json
import os
import asyncio
import base64
import logging
import copy # Import the copy module
from typing import Optional

from openai import OpenAI
from ..llm_provider import LLMProvider

# Set up logger
logger = logging.getLogger("droidrun")

class GeminiProvider(LLMProvider):
    """Gemini provider implementation."""

    def _initialize_client(self) -> None:
        """Initialize the Gemini client."""
        self.log_filename = 'messages_log.json' # Define log filename during init

        # Get API key from env var if not provided
        self.api_key = self.api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key not provided and not found in environment")

        # Set default model if not specified
        if not self.model_name:
            self.model_name = "gemini-1.5-flash" # Using a common Gemini model name

        # Initialize client with Gemini configuration
        # Ensure the base_url is correct for accessing Gemini via OpenAI-compatible endpoint
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta" # Adjust if using OpenAI library requires a specific format
            # Note: The OpenAI library might need specific configuration or a different library
            # might be preferred for direct Gemini access (e.g., google.generativeai).
            # Assuming the OpenAI library setup with base_url works for this context.
        )
        logger.info(f"Initialized Gemini client with model {self.model_name}")

    def _log_conversation_state(self, messages_to_log: list):
        """Logs the provided message list to the JSON log file."""
        try:
            log_data = []
            # Check if the log file exists and is not empty
            if os.path.exists(self.log_filename) and os.path.getsize(self.log_filename) > 0:
                try:
                    with open(self.log_filename, 'r', encoding='utf-8') as f:
                        log_data = json.load(f)
                    # Ensure it's a list; if not, reinitialize
                    if not isinstance(log_data, list):
                        logger.warning(f"Log file '{self.log_filename}' does not contain a valid JSON list. Reinitializing.")
                        log_data = []
                except json.JSONDecodeError:
                    logger.error(f"Could not decode JSON from '{self.log_filename}'. File might be corrupted. Reinitializing.")
                    # Optional: backup corrupted file
                    # try:
                    #     os.rename(self.log_filename, self.log_filename + f".corrupted_{int(time.time())}")
                    # except OSError as backup_error:
                    #     logger.error(f"Could not back up corrupted file: {backup_error}")
                    log_data = [] # Start with an empty list if decoding fails
                except Exception as read_err:
                     logger.error(f"Error reading log file '{self.log_filename}': {read_err}")
                     log_data = [] # Attempt to recover by reinitializing

            # Append the current messages list (it's already a copy passed to this method)
            log_data.append(messages_to_log)

            # Write the updated list back to the file
            with open(self.log_filename, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=4, ensure_ascii=False)

        except IOError as e:
            logger.error(f"An error occurred during file I/O for logging: {e}")
        except Exception as e:
            # Log unexpected errors during the logging process itself
            logger.error(f"An unexpected error occurred during logging: {e}", exc_info=True)


    async def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        screenshot_data: Optional[bytes] = None
    ) -> str:
        """
        Generate a response using Gemini, log the interaction, and return the response content.
        """
        try:
            # 1. Construct the messages list to send to the API
            messages_for_api = [
                {"role": "system", "content": system_prompt},
            ]

            # Add screenshot if provided
            if screenshot_data:
                base64_image = base64.b64encode(screenshot_data).decode('utf-8')
                # Gemini API expects image content differently than OpenAI vision
                # Adjust structure based on actual Gemini API requirements if not using OpenAI library compatibility layer
                # Assuming OpenAI compatible structure for now:
                messages_for_api.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here's the current screenshot of the device. Please analyze it to help with the next action."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                })
            else:
                 # If no screenshot, the user prompt is simpler
                 messages_for_api.append({"role": "user", "content": user_prompt})


            # If there was a screenshot, add the user text prompt as a separate message
            # This separation is often better for multimodal models.
            if screenshot_data:
                 messages_for_api.append({"role": "user", "content": user_prompt})


            # 2. Call the API
            logger.info(f"Sending request to Gemini model {self.model_name}...")
            # Ensure the API call structure matches the library/endpoint being used
            # The following assumes the OpenAI library is correctly configured for Gemini
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model_name,
                messages=messages_for_api, # Send the constructed list
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                # response_format might not be supported or named differently for Gemini
                # Check Gemini API documentation if using directly or compatibility layer limitations
                # response_format={"type": "json_object"} # Keep if supported and needed
            )
            logger.info("Received response from Gemini.")

            # 3. Process the response
            # Accessing response might differ based on actual API/library structure
            assistant_response_content = response.choices[0].message.content
            if not isinstance(assistant_response_content, str):
                 logger.warning(f"Assistant response content is not a string: {type(assistant_response_content)}. Converting to string.")
                 assistant_response_content = str(assistant_response_content)


            # 4. Update token usage statistics (if usage data is available and structured as expected)
            if hasattr(response, 'usage') and response.usage:
                usage = response.usage
                prompt_tokens = usage.prompt_tokens or 0
                completion_tokens = usage.completion_tokens or 0
                total_tokens = usage.total_tokens or (prompt_tokens + completion_tokens)

                self.update_token_usage(prompt_tokens, completion_tokens)

                # Log token usage
                logger.info("===== Token Usage Statistics =====")
                logger.info(f"API Call #{self.api_calls}")
                logger.info(f"This call: {prompt_tokens} prompt + {completion_tokens} completion = {total_tokens} tokens")
                logger.info(f"Cumulative: {self.get_token_usage_stats()}")
                logger.info("=================================")
            else:
                logger.warning("Token usage data not found in Gemini response.")


            # --- Logging the Full Exchange ---
            # Create a deep copy of the messages sent to the API *before* adding the response
            messages_full_exchange = copy.deepcopy(messages_for_api)
            # Append the assistant's response to this copy
            messages_full_exchange.append({"role": "assistant", "content": assistant_response_content})
            # Log this complete exchange state using the helper method
            self._log_conversation_state(messages_full_exchange)
            # --- End Logging ---


            # 5. Return the assistant's response content
            return assistant_response_content

        except Exception as e:
            logger.error(f"Error during Gemini API call or processing: {e}", exc_info=True)
            # Optionally log the request that failed
            try:
                messages_on_error = copy.deepcopy(messages_for_api) # Use the list prepared for the API
                messages_on_error.append({"role": "error", "content": f"Failed API call: {str(e)}"})
                self._log_conversation_state(messages_on_error)
            except Exception as log_err:
                 logger.error(f"Failed to log error state: {log_err}", exc_info=True)
            raise # Re-raise the original exception