"""
Ollama integration module for managing and interacting with local LLM models.
"""

import sys
import os
import platform
import requests
import streamlit as st
import subprocess
import backoff
from dotenv import load_dotenv
from llms.base import LLM
from constants import OLLAMA_MODEL_OPTIONS

# Load environment variables
load_dotenv()

# Get Ollama endpoint from environment or use default
OLLAMA_ENDPOINT = os.getenv('OLLAMA_ENDPOINT') or "http://localhost:11434"


class LocalLlms(LLM):
    """
    Class for interacting with local LLM models via Ollama.
    """
    
    def __init__(self, model_name, position_noti="content"):
        """
        Initialize the LocalLlms instance.
        
        Args:
            model_name (str): Name of the model to use
            position_noti (str): Where to display notifications ("content" or "sidebar")
        """
        self.model_name = model_name
        self.base_url = os.getenv('OLLAMA_ENDPOINT') or "http://localhost:11434"
        self.position_noti = position_noti
        self.pull_model()

    def pull_model(self):
        """
        Pull the specified model from the Ollama server.
        
        Raises:
            Exception: If model pull fails
        """
        st.spinner(f"Pulling model {self.model_name}...")
        
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"model": self.model_name}
            )

            if response.status_code != 200:
                error_msg = f"Failed to pull model {self.model_name}: {response.text}"
                self._show_notification(error_msg, "error")
                raise Exception(f"Model pull failed: {response.text}")

            self._show_notification(f"Model {self.model_name} pulled successfully.", "success")
            
        except Exception as e:
            self._show_notification(f"Error pulling model: {str(e)}", "error")
            raise

    def chat(self, messages, options=None):
        """
        Send messages to the model and return the assistant's response.
        
        Args:
            messages (list): List of message objects with role and content
            options (dict, optional): Additional options for the model
            
        Returns:
            dict: Response information including content and metadata
        """
        try:
            data = {
                "model": self.model_name, 
                "messages": messages,  
                "stream": False,       
            }
            
            if options:
                data["options"] = options

            response = requests.post(
                f"{self.base_url}/api/chat",
                json=data
            )
            
            if response.status_code == 200:
                response_json = response.json()
                assistant_message = response_json.get('message', {}).get('content', '')
                
                return {
                    "content": assistant_message,
                    "model": response_json.get('model'),
                    "created_at": response_json.get('created_at'),
                    "total_duration": response_json.get('total_duration'),
                    "load_duration": response_json.get('load_duration'),
                    "prompt_eval_count": response_json.get('prompt_eval_count'),
                    "prompt_eval_duration": response_json.get('prompt_eval_duration'),
                    "eval_count": response_json.get('eval_count'),
                    "eval_duration": response_json.get('eval_duration'),
                    "done": response_json.get('done')
                }
            else:
                print(f"Error: Received status code {response.status_code}")
                return None

        except Exception as e:
            print(f"Error: {e}")
            return None

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def create_agentic_chunker_message(self, system_prompt, messages, max_tokens=1000, temperature=1):
        """
        Create a message using the agentic chunker with automatic retry.
        
        Args:
            system_prompt (str): System prompt to guide the model
            messages (list): List of message objects
            max_tokens (int): Maximum number of tokens for context
            temperature (float): Temperature parameter for generation
            
        Returns:
            str: Generated content from the model
            
        Raises:
            Exception: Propagates exceptions after retries are exhausted
        """
        try:
            ollama_messages = [
                {"role": "system", "content": system_prompt}
            ] + messages

            response = self.chat(
                ollama_messages, 
                {"temperature": temperature, "num_ctx": max_tokens}
            )
            
            return response.get("content")
            
        except Exception as e:
            print(f"Error occurred: {e}, retrying...")
            raise e
        
    def generate_content(self, prompt):
        """
        Generate content from a prompt using the model.
        
        Args:
            prompt (str): The prompt to generate content from
            
        Returns:
            str: Generated content or empty string on error
        """
        try:
            data = {
                "model": self.model_name, 
                "prompt": prompt,
                "stream": False,       
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=data
            )

            if response.status_code == 200:
                response_json = response.json()
                return response_json.get("response", "")
            else:
                print(f"Error: Received status code {response.status_code}")
                return ""
                
        except Exception as e:
            print(f"Error generating content: {e}")
            return ""

    def _show_notification(self, message, notification_type):
        """
        Show a notification in the Streamlit UI.
        
        Args:
            message (str): Message to display
            notification_type (str): Type of notification (success, error, info)
        """
        if self.position_noti == "content":
            getattr(st, notification_type)(message)
        else:
            getattr(st.sidebar, notification_type)(message)


class OllamaManager:
    """
    Class for managing Ollama container and models.
    """
    
    def __init__(self):
        """Initialize the Ollama manager."""
        self.ollama_endpoint = os.getenv('OLLAMA_ENDPOINT') or "http://localhost:11434"

    def install_nvidia_toolkit(self):
        """Install NVIDIA Container Toolkit for GPU support."""
        st.info("Installing NVIDIA Container Toolkit...")
        
        commands = [
            "curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg",
            "curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list",
            "sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit",
            "sudo nvidia-ctk runtime configure --runtime=docker",
            "sudo systemctl restart docker"
        ]
        
        for cmd in commands:
            os.system(cmd)

    def has_nvidia_gpu(self):
        """
        Check if NVIDIA GPU is available.
        
        Returns:
            bool: True if NVIDIA GPU is available, False otherwise
        """
        try:
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def has_amd_gpu(self):
        """
        Check if AMD GPU is available.
        
        Returns:
            bool: True if AMD GPU is available, False otherwise
        """
        try:
            result = subprocess.run(['lspci'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return 'AMD' in result.stdout.decode()
        except FileNotFoundError:
            return False

    def remove_running_container(self, container_name, position_noti="content"):
        """
        Remove a running Docker container.
        
        Args:
            container_name (str): Name of the container to remove
            position_noti (str): Where to display notifications
        """
        # Check if the container is running
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", f"name={container_name}"], 
            capture_output=True, 
            text=True
        )
        
        if result.stdout.strip():  # Container is running
            os.system(f"docker rm -f {container_name}")
            self._show_notification(
                f"Removed the running container '{container_name}'.",
                "success",
                position_noti
            )

    def run_ollama_container(self, position_noti="content"):
        """
        Run the Ollama container based on available hardware.
        
        Args:
            position_noti (str): Where to display notifications
        """
        system = platform.system().lower()
        container_name = "ollama"

        # Remove the container if it's already running
        self.remove_running_container(container_name, position_noti)

        if system in ("linux", "darwin"):  # macOS or Linux
            if self.has_nvidia_gpu():
                self._show_notification(
                    "NVIDIA GPU detected. Installing NVIDIA Container Toolkit if necessary...",
                    "info",
                    position_noti
                )
                self.install_nvidia_toolkit()
                
                # Run Ollama container with NVIDIA GPU
                os.system(f"docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name {container_name} ollama/ollama")
                self._show_notification(
                    "Ollama container running with NVIDIA GPU!",
                    "success",
                    position_noti
                )
                
            elif self.has_amd_gpu():
                self._show_notification(
                    "AMD GPU detected. Starting Ollama with ROCm support...",
                    "info",
                    position_noti
                )
                
                # Run Ollama container with AMD GPU
                os.system(f"docker run -d --device /dev/kfd --device /dev/dri -v ollama:/root/.ollama -p 11434:11434 --name {container_name} ollama/ollama:rocm")
                self._show_notification(
                    "Ollama container running with AMD GPU!",
                    "success",
                    position_noti
                )
                
            else:
                self._show_notification(
                    "No GPU detected. Starting Ollama with CPU-only support...",
                    "info",
                    position_noti
                )
                
                # Run Ollama container with CPU-only
                os.system(f"docker run -d -v ollama:/root/.ollama -p 11434:11434 --name {container_name} ollama/ollama")
                self._show_notification(
                    "Ollama container running with CPU-only!",
                    "success",
                    position_noti
                )

        elif system == "windows":
            self._show_notification(
                "Please download and install Docker Desktop for Windows and run the following command manually:",
                "warning",
                position_noti
            )
            st.code(f"docker run -d -v ollama:/root/.ollama -p 11434:11434 --name {container_name} ollama/ollama")
    
    def run_ollama_model(self, model_name="gemma2:2b", position_noti="content"):
        """
        Run an Ollama model and return a LocalLlms instance.
        
        Args:
            model_name (str): Name of the model to run
            position_noti (str): Where to display notifications
            
        Returns:
            LocalLlms: Instance of LocalLlms or None if server not running
        """
        try:
            response = requests.get(self.ollama_endpoint)            
            if response.status_code != 200:
                self._show_notification(
                    "Ollama server is not running. Please start the server first.",
                    "error",
                    position_noti
                )
                return None
                
        except requests.ConnectionError:
            error_msg = "Ollama server is not reachable. Please check if it's running."
            self._show_notification(error_msg, "error", position_noti)
            
            if position_noti == "content":
                st.error(f"Model Name: {model_name}")
                st.error(f"position_noti: {position_noti}")
                
            return None
            
        # Create and return an instance of LocalLlms
        return LocalLlms(
            model_name,
            position_noti=position_noti
        )
        
    def _show_notification(self, message, notification_type, position):
        """
        Show a notification in the Streamlit UI.
        
        Args:
            message (str): Message to display
            notification_type (str): Type of notification (success, error, info, warning)
            position (str): Where to display notifications ("content" or "sidebar")
        """
        if position == "content":
            getattr(st, notification_type)(message)
        else:
            getattr(st.sidebar, notification_type)(message)