"""
UI Actions - Core UI interaction tools for Android device control.
"""

import os
import re
import json
import time
import tempfile
import asyncio
import aiofiles
import contextlib
from typing import Optional, Dict, Tuple, List, Any
from ..adb import Device, DeviceManager


class Tools:
    """Core UI interaction tools for Android device control."""

    def __init__(self, serial: str) -> None:
        # Instance‐level cache for clickable elements (index-based tapping)
        self.clickable_elements_cache: List[Dict[str, Any]] = []
        self.serial = serial
        self.device_manager = DeviceManager()
        self.last_screenshot = None

    def get_device_serial(self) -> str:
        """Get the device serial from environment variable."""
        return os.environ.get("DROIDRUN_DEVICE_SERIAL", "")

    async def get_device(self) -> Optional[Device]:
        """Get the device instance using the serial from environment variable.
        
        Returns:
            Device instance or None if not found
        """
        serial = self.get_device_serial()
        if not serial:
            raise ValueError("DROIDRUN_DEVICE_SERIAL environment variable not set")
            
        device = await self.device_manager.get_device(serial)
        if not device:
            raise ValueError(f"Device {serial} not found")
        
        return device

    def parse_package_list(self, output: str) -> List[Dict[str, str]]:
        """Parse the output of 'pm list packages -f' command.

        Args:
            output: Raw command output from 'pm list packages -f'

        Returns:
            List of dictionaries containing package info with 'package' and 'path' keys
        """
        apps = []
        for line in output.splitlines():
            if line.startswith("package:"):
                # Format is: "package:/path/to/base.apk=com.package.name"
                path_and_pkg = line[8:]  # Strip "package:"
                if "=" in path_and_pkg:
                    path, package = path_and_pkg.rsplit("=", 1)
                    apps.append({"package": package.strip(), "path": path.strip()})
        return apps

    async def get_clickables(self) -> list[dict[str, str | int | bool | None]]:
        """
        Get all clickable UI elements from the device using the custom TopViewService.
        
        This function interacts with the TopViewService app installed on the device
        to capture only the clickable UI elements. The service writes UI data
        to a JSON file on the device, which is then pulled to the host.
        
        Returns:
            List containing clickable UI elements extracted from the device screen
            Example: [{"text": "Settings", "className": "RelativeLayout", "index": 4, "bounds": "236,1850,439,2169", "type": "clickable", "isParent": True}, {"text": "icon_container", "className": "FrameLayout", "index": 16, "bounds": "240,1850,434,2038", "type": "text", "isParent": False, "parentIndex": 4}, {"text": "icon_title_container", "className": "FrameLayout", "index": 17, "bounds": "236,2038,439,2169", "type": "text", "isParent": False, "parentIndex": 4}]

        """
        global CLICKABLE_ELEMENTS_CACHE
        
        try:
            # Get the device
            if self.serial:
                device_manager = DeviceManager()
                device = await device_manager.get_device(self.serial)
                if not device:
                    raise ValueError(f"Device {self.serial} not found")
            else:
                device = await self.get_device()
            
            # Create a temporary file for the JSON
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
                local_path = temp.name
            
            try:
                # Clear logcat to make it easier to find our output
                await device._adb.shell(device._serial, "logcat -c")
                
                # Trigger the custom service via broadcast to get only interactive elements
                await device._adb.shell(device._serial, "am broadcast -a com.droidrun.portal.GET_ELEMENTS")
                
                # Poll for the JSON file path
                start_time = asyncio.get_event_loop().time()
                max_wait_time = 10  # Maximum wait time in seconds
                poll_interval = 0.2  # Check every 200ms
                
                device_path = None
                while asyncio.get_event_loop().time() - start_time < max_wait_time:
                    # Check logcat for the file path
                    logcat_output = await device._adb.shell(device._serial, "logcat -d | grep \"DROIDRUN_FILE\" | grep \"JSON data written to\" | tail -1")
                    
                    # Parse the file path if present
                    match = re.search(r"JSON data written to: (.*)", logcat_output)
                    if match:
                        device_path = match.group(1).strip()
                        break
                        
                    # Wait before polling again
                    await asyncio.sleep(poll_interval)
                
                # Check if we found the file path
                if not device_path:
                    raise ValueError(f"Failed to find the JSON file path in logcat after {max_wait_time} seconds")
                    
                # Pull the JSON file from the device
                await device._adb.pull_file(device._serial, device_path, local_path)
                
                # Read the JSON file
                async with aiofiles.open(local_path, "r", encoding="utf-8") as f:
                    json_content = await f.read()
                    
                # Clean up the temporary file
                with contextlib.suppress(OSError):
                    os.unlink(local_path)
                
                # Try to parse the JSON
                import json
                try:
                    ui_data = json.loads(json_content)
                    
                    # Process the JSON to extract elements
                    flattened_elements = []
                    
                    # Process the nested elements structure
                    if isinstance(ui_data, list):
                        # For each parent element in the list
                        for parent in ui_data:
                            # Add the parent if it's clickable (type should be 'clickable')
                            if parent.get('type') == 'clickable' and parent.get('index', -1) != -1:
                                parent_copy = {k: v for k, v in parent.items() if k != 'children'}
                                parent_copy['isParent'] = True
                                flattened_elements.append(parent_copy)
                            
                            # Process children 
                            children = parent.get('children', [])
                            for child in children:
                                # Add all children that have valid indices, regardless of type
                                # Include text elements as well, not just clickable ones
                                if child.get('index', -1) != -1:
                                    child_copy = child.copy()
                                    child_copy['isParent'] = False
                                    child_copy['parentIndex'] = parent.get('index')
                                    flattened_elements.append(child_copy)
                                    
                                    # Also process nested children if present
                                    nested_children = child.get('children', [])
                                    for nested_child in nested_children:
                                        if nested_child.get('index', -1) != -1:
                                            nested_copy = nested_child.copy()
                                            nested_copy['isParent'] = False
                                            nested_copy['parentIndex'] = child.get('index')
                                            nested_copy['grandparentIndex'] = parent.get('index')
                                            flattened_elements.append(nested_copy)
                    else:
                        # Old format handling (dictionary with clickable_elements)
                        clickable_elements = ui_data.get("clickable_elements", [])
                        for element in clickable_elements:
                            if element.get('index', -1) != -1:
                                element_copy = {k: v for k, v in element.items() if k != 'isClickable'}
                                flattened_elements.append(element_copy)
                    
                    # Update the global cache with the processed elements
                    CLICKABLE_ELEMENTS_CACHE = flattened_elements
                    
                    # Sort by index
                    flattened_elements.sort(key=lambda x: x.get('index', 0))
                    
                    # Create a summary of important text elements for each clickable parent
                    text_summary = []
                    parent_texts = {}
                    tappable_elements = []
                    
                    # Group text elements by their parent and identify tappable elements
                    for elem in flattened_elements:
                        # Track elements that are actually tappable (have bounds and either type clickable or are parents)
                        if elem.get('bounds') and (elem.get('type') == 'clickable' or elem.get('isParent')):
                            tappable_elements.append(elem.get('index'))
                        
                        if elem.get('type') == 'text' and elem.get('text'):
                            parent_id = elem.get('parentIndex')
                            if parent_id is not None:
                                if parent_id not in parent_texts:
                                    parent_texts[parent_id] = []
                                parent_texts[parent_id].append(elem.get('text'))
                    
                    # Create a text summary for parents with text children
                    for parent_id, texts in parent_texts.items():
                        # Find the parent element
                        parent = None
                        for elem in flattened_elements:
                            if elem.get('index') == parent_id:
                                parent = elem
                                break
                        
                        if parent:
                            # Mark if this element is directly tappable
                            tappable_marker = "🔘" if parent_id in tappable_elements else "📄"
                            summary = f"{tappable_marker} Element {parent_id} ({parent.get('className', 'Unknown')}): " + " | ".join(texts)
                            text_summary.append(summary)
                    
                    # Sort the text summary for better readability
                    text_summary.sort()
                    
                    # Count how many elements are actually tappable
                    tappable_count = len(tappable_elements)
                    
                    # Add a short sleep to ensure UI is fully loaded/processed
                    await asyncio.sleep(0.5)  # 500ms sleep
                    

                    clickable = flattened_elements
                    return clickable
                except json.JSONDecodeError:
                    raise ValueError("Failed to parse UI elements JSON data")
                
            except Exception as e:
                # Clean up in case of error
                with contextlib.suppress(OSError):
                    os.unlink(local_path)
                raise ValueError(f"Error retrieving clickable elements: {e}")
                
        except Exception as e:
            try:
                # Get the device
                if self.serial:
                    device_manager = DeviceManager()
                    device = await device_manager.get_device(self.serial)
                    if not device:
                        raise ValueError(f"Device {self.serial} not found")
                else:
                    device = await self.get_device()
                
                # Create a temporary file for the JSON
                with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
                    local_path = temp.name
                
                try:
                    # Clear logcat to make it easier to find our output
                    await device._adb.shell(device._serial, "logcat -c")
                    
                    # Trigger the custom service via broadcast to get only interactive elements
                    await device._adb.shell(device._serial, "am broadcast -a com.droidrun.portal.GET_ELEMENTS")
                    
                    # Poll for the JSON file path
                    start_time = asyncio.get_event_loop().time()
                    max_wait_time = 10  # Maximum wait time in seconds
                    poll_interval = 0.2  # Check every 200ms
                    
                    device_path = None
                    while asyncio.get_event_loop().time() - start_time < max_wait_time:
                        # Check logcat for the file path
                        logcat_output = await device._adb.shell(device._serial, "logcat -d | grep \"DROIDRUN_FILE\" | grep \"JSON data written to\" | tail -1")
                        
                        # Parse the file path if present
                        match = re.search(r"JSON data written to: (.*)", logcat_output)
                        if match:
                            device_path = match.group(1).strip()
                            break
                            
                        # Wait before polling again
                        await asyncio.sleep(poll_interval)
                    
                    # Check if we found the file path
                    if not device_path:
                        raise ValueError(f"Failed to find the JSON file path in logcat after {max_wait_time} seconds")
                        
                    # Pull the JSON file from the device
                    await device._adb.pull_file(device._serial, device_path, local_path)
                    
                    # Read the JSON file
                    async with aiofiles.open(local_path, "r", encoding="utf-8") as f:
                        json_content = await f.read()
                        
                    # Clean up the temporary file
                    with contextlib.suppress(OSError):
                        os.unlink(local_path)
                    
                    # Try to parse the JSON
                    import json
                    try:
                        ui_data = json.loads(json_content)
                        
                        # Process the JSON to extract elements
                        flattened_elements = []
                        
                        # Process the nested elements structure
                        if isinstance(ui_data, list):
                            # For each parent element in the list
                            for parent in ui_data:
                                # Add the parent if it's clickable (type should be 'clickable')
                                if parent.get('type') == 'clickable' and parent.get('index', -1) != -1:
                                    parent_copy = {k: v for k, v in parent.items() if k != 'children'}
                                    parent_copy['isParent'] = True
                                    flattened_elements.append(parent_copy)
                                
                                # Process children 
                                children = parent.get('children', [])
                                for child in children:
                                    # Add all children that have valid indices, regardless of type
                                    # Include text elements as well, not just clickable ones
                                    if child.get('index', -1) != -1:
                                        child_copy = child.copy()
                                        child_copy['isParent'] = False
                                        child_copy['parentIndex'] = parent.get('index')
                                        flattened_elements.append(child_copy)
                                        
                                        # Also process nested children if present
                                        nested_children = child.get('children', [])
                                        for nested_child in nested_children:
                                            if nested_child.get('index', -1) != -1:
                                                nested_copy = nested_child.copy()
                                                nested_copy['isParent'] = False
                                                nested_copy['parentIndex'] = child.get('index')
                                                nested_copy['grandparentIndex'] = parent.get('index')
                                                flattened_elements.append(nested_copy)
                        else:
                            # Old format handling (dictionary with clickable_elements)
                            clickable_elements = ui_data.get("clickable_elements", [])
                            for element in clickable_elements:
                                if element.get('index', -1) != -1:
                                    element_copy = {k: v for k, v in element.items() if k != 'isClickable'}
                                    flattened_elements.append(element_copy)
                        
                        # Update the global cache with the processed elements
                        CLICKABLE_ELEMENTS_CACHE = flattened_elements
                        
                        # Sort by index
                        flattened_elements.sort(key=lambda x: x.get('index', 0))
                        
                        # Create a summary of important text elements for each clickable parent
                        text_summary = []
                        parent_texts = {}
                        tappable_elements = []
                        
                        # Group text elements by their parent and identify tappable elements
                        for elem in flattened_elements:
                            # Track elements that are actually tappable (have bounds and either type clickable or are parents)
                            if elem.get('bounds') and (elem.get('type') == 'clickable' or elem.get('isParent')):
                                tappable_elements.append(elem.get('index'))
                            
                            if elem.get('type') == 'text' and elem.get('text'):
                                parent_id = elem.get('parentIndex')
                                if parent_id is not None:
                                    if parent_id not in parent_texts:
                                        parent_texts[parent_id] = []
                                    parent_texts[parent_id].append(elem.get('text'))
                        
                        # Create a text summary for parents with text children
                        for parent_id, texts in parent_texts.items():
                            # Find the parent element
                            parent = None
                            for elem in flattened_elements:
                                if elem.get('index') == parent_id:
                                    parent = elem
                                    break
                            
                            if parent:
                                # Mark if this element is directly tappable
                                tappable_marker = "🔘" if parent_id in tappable_elements else "📄"
                                summary = f"{tappable_marker} Element {parent_id} ({parent.get('className', 'Unknown')}): " + " | ".join(texts)
                                text_summary.append(summary)
                        
                        # Sort the text summary for better readability
                        text_summary.sort()
                        
                        # Count how many elements are actually tappable
                        tappable_count = len(tappable_elements)
                        
                        # Add a short sleep to ensure UI is fully loaded/processed
                        await asyncio.sleep(0.5)  # 500ms sleep
                        

                        clickable = flattened_elements
                        return clickable
                    except json.JSONDecodeError:
                        raise ValueError("Failed to parse UI elements JSON data")
                    
                except Exception as e:
                    # Clean up in case of error
                    with contextlib.suppress(OSError):
                        os.unlink(local_path)
                    raise ValueError(f"Error retrieving clickable elements: {e}")
                    
            except Exception as e:
                raise ValueError(f"Error getting clickable elements: {e}")

    async def tap_by_index(self, index: int) -> str:
        """
        Tap on a UI element by its index.
        
        This function uses the cached clickable elements from the last get_clickables call
        to find the element with the given index and tap on its center coordinates.
        
        Args:
            index: Index of the element to tap
        
        Returns:
            Result message
        """
        global CLICKABLE_ELEMENTS_CACHE
        
        try:
            # Check if we have cached elements
            if not CLICKABLE_ELEMENTS_CACHE:
                return "Error: No UI elements cached. Call get_clickables first."
            
            # Find the element with the given index
            element = None
            for item in CLICKABLE_ELEMENTS_CACHE:
                if item.get('index') == index:
                    element = item
                    break
            
            if not element:
                # List available indices to help the user
                indices = sorted([item.get('index') for item in CLICKABLE_ELEMENTS_CACHE if item.get('index') is not None])
                indices_str = ", ".join(str(idx) for idx in indices[:20])
                if len(indices) > 20:
                    indices_str += f"... and {len(indices) - 20} more"
                    
                return f"Error: No element found with index {index}. Available indices: {indices_str}"
            
            # Get the bounds of the element
            bounds_str = element.get('bounds')
            if not bounds_str:
                element_text = element.get('text', 'No text')
                element_type = element.get('type', 'unknown')
                element_class = element.get('className', 'Unknown class')
                
                # Check if this is a child element with a parent that can be tapped instead
                parent_suggestion = ""
                if 'parentIndex' in element:
                    parent_idx = element.get('parentIndex')
                    parent_suggestion = f" You might want to tap its parent element with index {parent_idx} instead."
                    
                return f"Error: Element with index {index} ('{element_text}', {element_class}, type: {element_type}) has no bounds and cannot be tapped directly.{parent_suggestion}"
            
            # Parse the bounds (format: "left,top,right,bottom")
            try:
                left, top, right, bottom = map(int, bounds_str.split(','))
            except ValueError:
                return f"Error: Invalid bounds format for element with index {index}: {bounds_str}"
            
            # Calculate the center of the element
            x = (left + right) // 2
            y = (top + bottom) // 2
            
            # Get the device and tap at the coordinates
            if self.serial:
                device_manager = DeviceManager()
                device = await device_manager.get_device(self.serial)
                if not device:
                    return f"Error: Device {self.serial} not found"
            else:
                device = await self.get_device()
            
            await device.tap(x, y)
            
            # Gather element details for the response
            element_text = element.get('text', 'No text')
            element_class = element.get('className', 'Unknown class')
            element_type = element.get('type', 'unknown')
            is_parent = element.get('isParent', False)
            
            # Create a descriptive response
            response_parts = []
            response_parts.append(f"Tapped element with index {index}")
            response_parts.append(f"Text: '{element_text}'")
            response_parts.append(f"Class: {element_class}")
            response_parts.append(f"Type: {element_type}")
            response_parts.append(f"Role: {'parent' if is_parent else 'child'}")
            
            # If it's a parent element, include information about its text children
            if is_parent:
                # Find all child elements that are text elements
                text_children = []
                for item in CLICKABLE_ELEMENTS_CACHE:
                    if (item.get('parentIndex') == index and 
                        item.get('type') == 'text' and 
                        item.get('text')):
                        text_children.append(item.get('text'))
                
                if text_children:
                    response_parts.append(f"Contains text: {' | '.join(text_children)}")
            
            # If it's a child element, include parent information
            if not is_parent and 'parentIndex' in element:
                parent_index = element.get('parentIndex')
                # Find the parent element
                parent = None
                for item in CLICKABLE_ELEMENTS_CACHE:
                    if item.get('index') == parent_index:
                        parent = item
                        break
                
                if parent:
                    parent_text = parent.get('text', 'No text')
                    response_parts.append(f"Parent: {parent_index} ('{parent_text}')")
                    
                    # Find sibling text elements (other children of the same parent)
                    sibling_texts = []
                    for item in CLICKABLE_ELEMENTS_CACHE:
                        if (item.get('parentIndex') == parent_index and 
                            item.get('index') != index and
                            item.get('type') == 'text' and 
                            item.get('text')):
                            sibling_texts.append(item.get('text'))
                    
                    if sibling_texts:
                        response_parts.append(f"Related text: {' | '.join(sibling_texts)}")
            
            response_parts.append(f"Coordinates: ({x}, {y})")
            
            return " | ".join(response_parts)
        except ValueError as e:
            return f"Error: {str(e)}"

    # Rename the old tap function to tap_by_coordinates for backward compatibility
    async def tap_by_coordinates(self, x: int, y: int) -> bool:
        """
        Tap on the device screen at specific coordinates. 
        
        Args:
            x: X coordinate
            y: Y coordinate
        
        Returns:
            Bool indicating success or failure
        """
        try:
            if self.serial:
                device_manager = DeviceManager()
                device = await device_manager.get_device(self.serial)
                if not device:
                    return f"Error: Device {self.serial} not found"
            else:
                device = await self.get_device()
            
            await device.tap(x, y)
            print(f"Tapped at coordinates ({x}, {y})")
            return True
        except ValueError as e:
            print(f"Error: {str(e)}")
            return False

    # Replace the old tap function with the new one
    async def tap(self, index: int) -> str:
        """
        Tap on a UI element by its index.
        
        This function uses the cached clickable elements from the last get_clickables call
        to find the element with the given index and tap on its center coordinates.
        
        Args:
            index: Index of the element to tap
        
        Returns:
            Result message
        """
        return await self.tap_by_index(index)

    async def swipe(
        self,
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration_ms: int = 300
    ) -> bool:
        """
        Performs a straight-line swipe gesture on the device screen.
        To perform a hold (long press), set the start and end coordinates to the same values and increase the duration as needed.        
        Args:
            start_x: Starting X coordinate
            start_y: Starting Y coordinate
            end_x: Ending X coordinate
            end_y: Ending Y coordinate
            duration_ms: Duration of swipe in milliseconds
        Returns:
            Bool indicating success or failure
        """
        try:
            if self.serial:
                device_manager = DeviceManager()
                device = await device_manager.get_device(self.serial)
                if not device:
                    return f"Error: Device {self.serial} not found"
            else:
                device = await self.get_device()
            
            await device.swipe(start_x, start_y, end_x, end_y, duration_ms)
            return True
        except ValueError as e:
            print(f"Error: {str(e)}")
            return False

    async def input_text(self, text: str) -> str:
        """
        Input text on the device.
        
        Args:
            text: Text to input. Can contain spaces and special characters.
        """
        try:
            if self.serial:
                device_manager = DeviceManager()
                device = await device_manager.get_device(self.serial)
                if not device:
                    return f"Error: Device {self.serial} not found"
            else:
                device = await self.get_device()
            
            # Function to escape special characters
            def escape_text(s: str) -> str:
                # Escape special characters that need shell escaping, excluding space
                special_chars = '[]()|&;$<>\\`"\'{}#!?^~'  # Removed space from special chars
                escaped = ''
                for c in s:
                    if c == ' ':
                        escaped += ' '  # Just add space without escaping
                    elif c in special_chars:
                        escaped += '\\' + c
                    else:
                        escaped += c
                return escaped
            
            # Split text into smaller chunks (max 500 chars)
            chunk_size = 500
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
            
            for chunk in chunks:
                # Escape the text chunk
                escaped_chunk = escape_text(chunk)
                
                # Try different input methods if one fails
                methods = [
                    f'input text "{escaped_chunk}"',  # Standard method
                    f'am broadcast -a ADB_INPUT_TEXT --es msg "{escaped_chunk}"',  # Broadcast intent method
                    f'input keyboard text "{escaped_chunk}"'  # Keyboard method
                ]
                
                success = False
                last_error = None
                
                for method in methods:
                    try:
                        await device._adb.shell(device._serial, method)
                        success = True
                        break
                    except Exception as e:
                        last_error = str(e)
                        continue
                
                if not success:
                    return f"Error: Failed to input text chunk. Last error: {last_error}"
                
                # Small delay between chunks
                await asyncio.sleep(0.1)
            
            return f"Text input completed: {text}"
        except ValueError as e:
            return f"Error: {str(e)}"

    async def press_key(self, keycode: int) -> str:
        """
        Press a key on the device.
        
        Common keycodes:
        - 3: HOME
        - 4: BACK
        - 24: VOLUME UP
        - 25: VOLUME DOWN
        - 26: POWER
        - 82: MENU
        
        Args:
            keycode: Android keycode to press
        """
        try:
            if self.serial:
                device_manager = DeviceManager()
                device = await device_manager.get_device(self.serial)
                if not device:
                    return f"Error: Device {self.serial} not found"
            else:
                device = await self.get_device()
            
            key_names = {
                3: "HOME",
                4: "BACK",
                24: "VOLUME UP",
                25: "VOLUME DOWN",
                26: "POWER",
                82: "MENU",
            }
            key_name = key_names.get(keycode, str(keycode))
            
            await device.press_key(keycode)
            return f"Pressed key {key_name}"
        except ValueError as e:
            return f"Error: {str(e)}"

    async def start_app(
        self,
        package: str,
        activity: str = ""
    ) -> str:
        """
        Start an app on the device.
        
        Args:
            package: Package name (e.g., "com.android.settings")
            activity: Optional activity name
        """
        try:
            if self.serial:
                device_manager = DeviceManager()
                device = await device_manager.get_device(self.serial)
                if not device:
                    return f"Error: Device {self.serial} not found"
            else:
                device = await self.get_device()
            
            result = await device.start_app(package, activity)
            return result
        except ValueError as e:
            return f"Error: {str(e)}"

    async def install_app(
        self,
        apk_path: str, 
        reinstall: bool = False, 
        grant_permissions: bool = True
    ) -> str:
        """
        Install an app on the device.
        
        Args:
            apk_path: Path to the APK file
            reinstall: Whether to reinstall if app exists
            grant_permissions: Whether to grant all permissions
        """
        try:
            if self.serial:
                device_manager = DeviceManager()
                device = await device_manager.get_device(self.serial)
                if not device:
                    return f"Error: Device {self.serial} not found"
            else:
                device = await self.get_device()
            
            if not os.path.exists(apk_path):
                return f"Error: APK file not found at {apk_path}"
            
            result = await device.install_app(apk_path, reinstall, grant_permissions)
            return result
        except ValueError as e:
            return f"Error: {str(e)}"

    async def uninstall_app(
        self,
        package: str, 
        keep_data: bool = False
    ) -> str:
        """
        Uninstall an app from the device.
        
        Args:
            package: Package name to uninstall
            keep_data: Whether to keep app data and cache
        """
        try:
            if self.serial:
                device_manager = DeviceManager()
                device = await device_manager.get_device(self.serial)
                if not device:
                    return f"Error: Device {self.serial} not found"
            else:
                device = await self.get_device()
            
            result = await device.uninstall_app(package, keep_data)
            return result
        except ValueError as e:
            return f"Error: {str(e)}"

    async def take_screenshot(self) -> bool:
        """
        Take a screenshot of the device.

        This function captures the current screen and adds the screenshot to context in the next message.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.serial:
                device_manager = DeviceManager()
                device = await device_manager.get_device(self.serial)
                if not device:
                    raise ValueError(f"Device {self.serial} not found")
            else:
                device = await self.get_device()
            screen_tuple = await device.take_screenshot()
            self.last_screenshot = screen_tuple[1]
            return True
        except ValueError as e:
            raise ValueError(f"Error taking screenshot: {str(e)}")

    async def list_packages(
        self,
        include_system_apps: bool = False
    ) -> List[str]:
        """
        List installed packages on the device.
        
        Args:
            include_system_apps: Whether to include system apps (default: False)
        
        Returns:
            List of package names
        """
        try:
            if self.serial:
                device_manager = DeviceManager()
                device = await device_manager.get_device(self.serial)
                if not device:
                    raise ValueError(f"Device {self.serial} not found")
            else:
                device = await self.get_device()
            
            # Use the direct ADB command to get packages with paths
            cmd = ["pm", "list", "packages", "-f"]
            if not include_system_apps:
                cmd.append("-3")
                
            output = await device._adb.shell(device._serial, " ".join(cmd))
            
            # Parse the package list using the function
            packages = self.parse_package_list(output)            
            # Format package list for better readability
            package_list = [pack["package"] for pack in packages]            
            return package_list
        except ValueError as e:
            raise ValueError(f"Error listing packages: {str(e)}")

    async def complete(self, result: str) -> str:
        """Complete the task with a result message.
        
        Args:
            result: The result message
            
        Returns:
            Success message
        """
        return f"Task completed: {result}"

    async def extract(self, filename: Optional[str] = None) -> str:
        """Extract and save the current UI state to a JSON file.
        
        This function captures the current UI state including all UI elements
        and saves it to a JSON file for later analysis or reference.
        
        Args:
            filename: Optional filename to save the UI state (defaults to ui_state_TIMESTAMP.json)
            
        Returns:
            Path to the saved JSON file
        """
        try:
            # Generate default filename if not provided
            if not filename:
                timestamp = int(time.time())
                filename = f"ui_state_{timestamp}.json"
            
            # Ensure the filename ends with .json
            if not filename.endswith(".json"):
                filename += ".json"
                
            # Get the UI elements
            ui_elements = await self.get_all_elements(self.serial)
            
            # Save to file
            save_path = os.path.abspath(filename)
            async with aiofiles.open(save_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(ui_elements, indent=2))
                
            return f"UI state extracted and saved to {save_path}"
            
        except Exception as e:
            return f"Error extracting UI state: {e}"

    async def get_all_elements(self) -> Dict[str, Any]:
        """
        Get all UI elements from the device, including non-interactive elements.
        
        This function interacts with the TopViewService app installed on the device
        to capture all UI elements, even those that are not interactive. This provides
        a complete view of the UI hierarchy for analysis or debugging purposes.
        
        Returns:
            Dictionary containing all UI elements extracted from the device screen
        """
        try:
            # Get the device
            device_manager = DeviceManager()
            device = await device_manager.get_device(self.serial)
            if not device:
                raise ValueError(f"Device {self.serial} not found")
            
            # Create a temporary file for the JSON
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
                local_path = temp.name
            
            try:
                # Clear logcat to make it easier to find our output
                await device._adb.shell(device._serial, "logcat -c")
                
                # Trigger the custom service via broadcast to get ALL elements
                await device._adb.shell(device._serial, "am broadcast -a com.droidrun.portal.GET_ALL_ELEMENTS")
                
                # Poll for the JSON file path
                start_time = asyncio.get_event_loop().time()
                max_wait_time = 10  # Maximum wait time in seconds
                poll_interval = 0.2  # Check every 200ms
                
                device_path = None
                while asyncio.get_event_loop().time() - start_time < max_wait_time:
                    # Check logcat for the file path
                    logcat_output = await device._adb.shell(device._serial, "logcat -d | grep \"DROIDRUN_FILE\" | grep \"JSON data written to\" | tail -1")
                    
                    # Parse the file path if present
                    match = re.search(r"JSON data written to: (.*)", logcat_output)
                    if match:
                        device_path = match.group(1).strip()
                        break
                        
                    # Wait before polling again
                    await asyncio.sleep(poll_interval)
                
                # Check if we found the file path
                if not device_path:
                    raise ValueError(f"Failed to find the JSON file path in logcat after {max_wait_time} seconds")
                    
                # Pull the JSON file from the device
                await device._adb.pull_file(device._serial, device_path, local_path)
                
                # Read the JSON file
                async with aiofiles.open(local_path, "r", encoding="utf-8") as f:
                    json_content = await f.read()
                    
                # Clean up the temporary file
                with contextlib.suppress(OSError):
                    os.unlink(local_path)
                
                # Try to parse the JSON
                import json
                try:
                    ui_data = json.loads(json_content)
                    
                    return {
                        "all_elements": ui_data,
                        "count": len(ui_data) if isinstance(ui_data, list) else sum(1 for _ in ui_data.get("elements", [])),
                        "message": "Retrieved all UI elements from the device screen"
                    }
                except json.JSONDecodeError:
                    raise ValueError("Failed to parse UI elements JSON data")
                
            except Exception as e:
                # Clean up in case of error
                with contextlib.suppress(OSError):
                    os.unlink(local_path)
                raise ValueError(f"Error retrieving all UI elements: {e}")
                
        except Exception as e:
            raise ValueError(f"Error getting all UI elements: {e}")