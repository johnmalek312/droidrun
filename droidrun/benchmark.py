#!/usr/bin/env python3
"""
Benchmark DroidRun agent using AndroidWorld tasks.

This script loads tasks from AndroidWorld and runs them using the DroidRun agent,
then evaluates the success using AndroidWorld's evaluation metrics.
"""

import os
import sys
import asyncio
import random
import argparse
import logging
import subprocess
import time
import json
from datetime import datetime
from typing import List, Type, Optional, Dict, Any

# Add paths to both packages
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DROIDRUN_PATH = os.path.join(SCRIPT_DIR, "droidrun")
sys.path.insert(0, DROIDRUN_PATH)

# Import from AndroidWorld
from android_world import registry
from agno.models.openai import OpenAIChat
from android_world.env import env_launcher



from android_world.task_evals import task_eval

# Import from local DroidRun
from droidrun.tools import DeviceManager
from droidrun.agent import LLMReasoner, ReActAgent

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")

class DroidRunBenchmark:
    """Benchmark DroidRun agent using AndroidWorld tasks."""
    
    def __init__(
        self,
        adb_path: str,
        console_port: int = 5554,
        perform_emulator_setup: bool = False,
        llm_model: str = "gpt-4-vision-preview",
        task_family: str = registry.TaskRegistry.ANDROID_FAMILY,
        specific_tasks: Optional[List[str]] = None,
        n_task_combinations: int = 1,
        random_seed: int =64532,
        results_dir: str = "benchmark_results",
        skip_first_n: int = 10
    ):
        """Initialize the benchmark.
        
        Args:
            adb_path: Path to ADB executable
            console_port: Emulator console port (usually 5554 for first emulator)
            perform_emulator_setup: Whether to set up the emulator (only needed once)
            llm_model: LLM model to use for the agent
            task_family: Task family to benchmark
            specific_tasks: List of specific tasks to run (if None, run all)
            n_task_combinations: Number of combinations to run for each task
            random_seed: Random seed for reproducibility
            results_dir: Directory to save benchmark results
            skip_first_n: Number of tasks to skip at the beginning
        """
        self.adb_path = adb_path
        self.console_port = console_port
        self.perform_emulator_setup = perform_emulator_setup
        self.llm_model = llm_model
        self.task_family = task_family
        self.specific_tasks = specific_tasks
        self.n_task_combinations = n_task_combinations
        self.random_seed = random_seed
        self.results_dir = results_dir
        self.skip_first_n = skip_first_n
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # Will be set during initialization
        self.env = None
        self.task_registry = None
        self.results = []
        
    async def initialize(self):
        """Initialize the environment and tasks."""
        logger.info("Initializing environment with ADB at %s", self.adb_path)
        
        # Initialize Android environment using AndroidWorld's setup
        self.env = env_launcher.load_and_setup_env(
            console_port=self.console_port,
            emulator_setup=self.perform_emulator_setup,
            adb_path=self.adb_path,
        )
        
        # Reset the environment to start from a clean state
        self.env.reset(go_home=True)
        
        # Enable accessibility service using direct shell command
        logger.info("Enabling DroidRun accessibility service...")

        # Get task registry from AndroidWorld
        self.task_registry = registry.TaskRegistry()
        
        # Create device manager for DroidRun
        self.device_manager = DeviceManager()
        
        logger.info("Environment initialized")
    
    def create_task_suite(self) -> List[task_eval.TaskEval]:
        """Create a suite of tasks to benchmark.
        
        Returns:
            List of task instances to evaluate
        """
        # Get registry for specific family
        task_dict = self.task_registry.get_registry(family=self.task_family)
        
        # Filter to specific tasks if requested
        if self.specific_tasks:
            filtered_dict = {}
            for task_name in self.specific_tasks:
                if task_name in task_dict:
                    filtered_dict[task_name] = task_dict[task_name]
                else:
                    logger.warning(f"Task {task_name} not found in {self.task_family}")
            task_dict = filtered_dict
        
        # Create task instances
        task_suite = []
        random.seed(self.random_seed)
        
        total_tasks = len(task_dict) * self.n_task_combinations
        logger.info(f"Creating task suite with {total_tasks} total tasks...")
        
        for task_name, task_class in task_dict.items():
            for i in range(self.n_task_combinations):
                # Generate random parameters for the task
                params = task_class.generate_random_params()
                # Add a seed for reproducibility
                params["seed"] = self.random_seed + i
                # Create task instance
                task_instance = task_class(params)
                task_suite.append(task_instance)
                
                logger.info(f"Created task: {task_name} (instance {i+1}/{self.n_task_combinations})")
                
        logger.info(f"Created task suite with {len(task_suite)} tasks")
        return task_suite
    
    async def create_agent(self, device_serial: str, task_goal: str, max_steps: int) -> ReActAgent:
        """Create a DroidRun agent for the given task.
        
        Args:
            device_serial: Device serial number
            task_goal: Goal for the agent to accomplish
            
        Returns:
            Configured ReActAgent
        """

        llm = LLMReasoner(
            llm_provider="gemini",
            model_name="gemini-2.0-flash",
            api_key="YOUR API KEY",
            temperature=1
        )


        agent = ReActAgent(
        task=task_goal,
        llm=llm,
        device_serial=device_serial,  # Will use first available device
        max_steps=max_steps
        )
        # Run the agent
        return agent
        
    
    def enable_accessibility_service(self, device_serial: str, disable_first: bool = False):
        """Enable the accessibility service using ADB commands."""
        try:
            if disable_first:
                # First disable all accessibility services
                disable_cmd = [
                    self.adb_path,
                    "-s", device_serial,
                    "shell",
                    "settings put secure enabled_accessibility_services ''"
                ]
                subprocess.run(disable_cmd, check=True)
                logger.info("Disabled all accessibility services")
                time.sleep(2)  # Small delay after disabling
            
            # Enable our specific service
            enable_service_cmd = [
                self.adb_path,
                "-s", device_serial,
                "shell",
                "settings put secure enabled_accessibility_services com.droidrun.portal/com.droidrun.portal.DroidrunPortalService"
            ]
            
            # Command to enable accessibility globally
            enable_global_cmd = [
                self.adb_path,
                "-s", device_serial,
                "shell",
                "settings put secure accessibility_enabled 1"
            ]
            
            # Execute commands
            subprocess.run(enable_service_cmd, check=True)
            subprocess.run(enable_global_cmd, check=True)
            logger.info("Successfully enabled accessibility service")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to enable accessibility service: {e}")
            return False

    def ensure_media_directories(self, device_serial: str):
        """Ensure required media directories exist on the device."""
        try:
            # List of essential directories
            directories = [
                "/sdcard/Pictures",
                "/sdcard/Movies",
                "/sdcard/DCIM",
                "/storage/emulated/0/Pictures",
                "/storage/emulated/0/Movies",
                "/storage/emulated/0/DCIM"
            ]
            
            for directory in directories:
                # Create directory command
                mkdir_cmd = [
                    self.adb_path,
                    "-s", device_serial,
                    "shell",
                    f"mkdir -p {directory}"
                ]
                # Set permissions command
                chmod_cmd = [
                    self.adb_path,
                    "-s", device_serial,
                    "shell",
                    f"chmod 777 {directory}"
                ]
                
                subprocess.run(mkdir_cmd, check=True)
                subprocess.run(chmod_cmd, check=True)
                
            logger.info("Ensured media directories exist")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create media directories: {e}")
            return False

    def save_task_result(self, task_result: Dict[str, Any]):
        """Save individual task result to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        task_name = task_result["task_name"]
        filename = f"task_result_{task_name}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        # Add metadata to the task result
        result_data = {
            "timestamp": timestamp,
            "task_family": self.task_family,
            "random_seed": self.random_seed,
            "result": {
                "task_name": task_result["task_name"],
                "goal": task_result["goal"],
                "success_score": task_result.get("success_score", 0),
                "success": task_result.get("success", False),
                "total_steps": task_result.get("total_steps", 0),
                "action_steps": task_result.get("action_steps", 0),
                "steps": task_result.get("steps", []),  # Include all steps with their types
                "error": task_result.get("error", None)  # Include error if present
            }
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(result_data, f, indent=2)
            logger.info(f"Task result saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save task result to {filepath}: {e}")

    async def run_benchmark(self):
        """Run the benchmark on all tasks and collect results."""
        await self.initialize()
        
        # Create task suite
        task_suite = self.create_task_suite()
        total_tasks = len(task_suite)
        logger.info(f"Created {total_tasks} task instances to benchmark")
        
        if self.skip_first_n > 0:
            if self.skip_first_n >= total_tasks:
                logger.error(f"Cannot skip {self.skip_first_n} tasks as only {total_tasks} tasks are available")
                return
                
            logger.info(f"Skipping first {self.skip_first_n} tasks out of {total_tasks} total tasks")
            task_suite = task_suite[self.skip_first_n:]
            logger.info(f"Remaining tasks to run: {len(task_suite)}")
            
            if not task_suite:
                logger.warning("No tasks remaining after skipping")
                return
        
        # Get device serial from first available device
        devices = await self.device_manager.list_devices()
        if not devices:
            logger.error("No devices found. Make sure emulator is running.")
            return
        
        device_serial = devices[0].serial
        logger.info(f"Using device: {device_serial}")
        # Set device serial as environment variable
        os.environ["DROIDRUN_DEVICE_SERIAL"] = device_serial
        logger.info(f"Set DROIDRUN_DEVICE_SERIAL environment variable to {device_serial}")

        # Run each task
        for task_instance in task_suite:
            task_name = task_instance.__class__.__name__
            logger.info(f"Running task: {task_name}")
            
            # Reset and enable accessibility service before each task
            self.enable_accessibility_service(device_serial, disable_first=True)
            
            # Ensure media directories exist
            self.ensure_media_directories(device_serial)
            
            # Initialize the task
            task_instance.initialize_task(self.env)
            goal = task_instance.goal

            logger.info(f"Task goal: {goal}")
            logger.info(f"Task complexity: {task_instance.complexity}")
            logger.info(f"Max Steps: {task_instance.complexity * 10}")
            
            # Create agent for this task
            agent = await self.create_agent(device_serial, goal, task_instance.complexity * 10)
            
            # Run the agent
            try:
                steps_and_count = await agent.run()
                steps = steps_and_count[0]  # Get the steps list
                action_step_count = steps_and_count[1]  # Get the actual action count
                
                # Check if task was successful
                success_score = task_instance.is_successful(self.env)
                
                # Create result
                result = {
                    "task_name": task_name,
                    "goal": goal,
                    "success_score": success_score,
                    "success": success_score == 1.0,
                    "total_steps": len(steps),  # Total steps including thoughts and observations
                    "action_steps": action_step_count,  # Only actual device interactions
                    "steps": [step.to_dict() for step in steps]  # Include the actual steps taken
                }
                
                self.results.append(result)
                # Save individual task result immediately
                self.save_task_result(result)
                
                logger.info(f"Task {task_name} completed with score {success_score}")
                logger.info(f"Total steps taken: {len(steps)}, Action steps: {action_step_count}")
                
            except Exception as e:
                logger.exception(f"Error running task {task_name}: {e}")
                error_result = {
                    "task_name": task_name,
                    "goal": goal,
                    "success_score": 0,
                    "success": False,
                    "error": str(e),
                    "total_steps": 0,
                    "action_steps": 0
                }
                self.results.append(error_result)
                # Save failed task result
                self.save_task_result(error_result)
            finally:
                # Clean up
                try:
                    task_instance.tear_down(self.env)
                except Exception as e:
                    logger.warning(f"Error during task teardown: {e}")
        
        # Print summary
        self.print_summary()
        
        # Close the environment
        self.env.close()
    
    def print_summary(self):
        """Print a summary of benchmark results."""
        if not self.results:
            logger.info("No results to summarize.")
            return
        
        total_tasks = len(self.results)
        successful_tasks = sum(1 for r in self.results if r.get("success", False))
        success_rate = successful_tasks / total_tasks * 100
        
        total_action_steps = sum(r.get("action_steps", 0) for r in self.results)
        total_steps = sum(r.get("total_steps", 0) for r in self.results)
        avg_action_steps = total_action_steps / total_tasks
        avg_total_steps = total_steps / total_tasks
        
        logger.info("=" * 50)
        logger.info(f"BENCHMARK SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total tasks: {total_tasks}")
        logger.info(f"Successful tasks: {successful_tasks}")
        logger.info(f"Success rate: {success_rate:.2f}%")
        logger.info(f"Average action steps per task: {avg_action_steps:.2f}")
        logger.info(f"Average total steps per task: {avg_total_steps:.2f}")
        logger.info("=" * 50)
        logger.info("Task details:")
        
        for i, result in enumerate(self.results):
            logger.info(f"{i+1}. {result['task_name']}: {'✓' if result.get('success', False) else '✗'}")
            logger.info(f"   Goal: {result['goal']}")
            logger.info(f"   Score: {result.get('success_score', 0)}")
            if "error" in result:
                logger.info(f"   Error: {result['error']}")
            logger.info(f"   Action Steps: {result.get('action_steps', 'N/A')}")
            logger.info(f"   Total Steps: {result.get('total_steps', 'N/A')}")
            logger.info("-" * 40)


async def main():
    parser = argparse.ArgumentParser(description="Benchmark DroidRun using AndroidWorld tasks")
    parser.add_argument("--adb-path", type=str, default=None, help="Path to ADB executable")
    parser.add_argument("--console-port", type=int, default=5554, help="Emulator console port")
    parser.add_argument("--setup-emulator", action="store_true", help="Perform emulator setup")
    parser.add_argument("--llm-model", type=str, default="gpt-4-vision-preview", help="LLM model to use")
    parser.add_argument("--task-family", type=str, default=registry.TaskRegistry.ANDROID_WORLD_FAMILY, 
                        help="Task family to benchmark")
    parser.add_argument("--tasks", type=str, nargs="*", help="Specific tasks to run")
    parser.add_argument("--combinations", type=int, default=1, help="Number of combinations per task")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--results-dir", type=str, default="benchmark_results",
                      help="Directory to save benchmark results")
    parser.add_argument("--skip-first", type=int, default=0,
                      help="Number of tasks to skip at the beginning")
    
    args = parser.parse_args()
    
    # Find ADB path if not specified
    if not args.adb_path:
        potential_paths = [
            os.path.expanduser('~/Library/Android/sdk/platform-tools/adb'),
            os.path.expanduser('~/Android/Sdk/platform-tools/adb'),
        ]
        for path in potential_paths:
            if os.path.isfile(path):
                args.adb_path = path
                break
        if not args.adb_path:
            parser.error("Could not find ADB. Please specify --adb-path.")
    
    benchmark = DroidRunBenchmark(
        adb_path=args.adb_path,
        console_port=args.console_port,
        perform_emulator_setup=args.setup_emulator,
        llm_model=args.llm_model,
        task_family=args.task_family,
        specific_tasks=args.tasks,
        n_task_combinations=args.combinations,
        random_seed=args.seed,
        results_dir=args.results_dir,
        skip_first_n=args.skip_first
    )
    
    await benchmark.run_benchmark()

if __name__ == "__main__":
    asyncio.run(main()) 