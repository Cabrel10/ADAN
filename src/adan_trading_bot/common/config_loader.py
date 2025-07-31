"""
Utility to load and resolve configuration with environment variables.
"""
import os
from typing import Dict, Any, Optional
import re
import yaml

class ConfigLoader:
    """Load and resolve configuration with environment variables."""
    
    @classmethod
    def resolve_env_vars(cls, config: Dict[str, Any], parent_key: str = '', processed: Optional[set] = None, root_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Recursively resolve environment variables and path references in the configuration.
        
        Args:
            config: The configuration dictionary to process
            parent_key: The parent key for nested dictionaries (used for error messages)
            processed: Set of already processed keys to detect circular references
            root_config: The root configuration dict (always use for dotted path resolution)
        Returns:
            Dict with resolved environment variables and paths
        Raises:
            ValueError: If there are undefined variables or circular references
        """
        if root_config is None:
            root_config = config
        if processed is None:
            processed = set()
            print("[DEBUG] Starting resolution of environment variables")
            
        # Make a deep copy to avoid modifying the original
        config_copy = {k: v for k, v in config.items()}
        
        # Detect circular references
        current_key = f"{parent_key}" if parent_key else "root"
        if current_key in processed:
            raise ValueError(f"Circular reference detected in config at {current_key}")
        processed.add(current_key)
        
        print(f"[DEBUG] Processing config section: {current_key}")
        print(f"[DEBUG] Current config keys: {list(config_copy.keys())}")
        
        # First pass: resolve all environment variables (${ENV_VAR} syntax)
        for key, value in list(config_copy.items()):
            if isinstance(value, str):
                # Handle environment variables like ${ENV_VAR}
                def replace_env_var(match):
                    env_var = match.group(1)
                    if env_var in os.environ:
                        return os.environ[env_var]
                    return match.group(0)  # Return original if not found
                
                new_value = re.sub(r'\${([A-Z0-9_]+)}', replace_env_var, value)
                if new_value != value:
                    config_copy[key] = new_value
                    print(f"[DEBUG] Resolved env var in {parent_key}.{key}: {value} -> {new_value}")
        
        # Second pass: resolve simple variables (non-nested)
        changed = True
        while changed:
            changed = False
            for key, value in list(config_copy.items()):
                if isinstance(value, dict):
                    # Process nested dictionaries recursively
                    config_copy[key] = cls.resolve_env_vars(value, f"{parent_key}.{key}" if parent_key else key, set(processed), root_config)
                elif isinstance(value, str) and '${' in value:
                    # Handle simple variable references (no dots)
                    def replace_simple_var(match):
                        var_name = match.group(1)
                        if '.' not in var_name:  # Only handle simple variables in this pass
                            try:
                                return str(config_copy[var_name])
                            except KeyError:
                                # Check if it's an environment variable that wasn't resolved earlier
                                if var_name in os.environ:
                                    return os.environ[var_name]
                                return match.group(0)  # Return original if not found
                        return match.group(0)
                    
                    new_value = re.sub(r'\${([^}]+)}', replace_simple_var, value)
                    if new_value != value:
                        config_copy[key] = new_value
                        changed = True
        
        # Third pass: resolve nested paths (with dots)
        for key, value in list(config_copy.items()):
            if isinstance(value, str) and '${' in value:
                print(f"[DEBUG] Resolving nested path in {parent_key}.{key}: {value}")
                
                def replace_nested_var(match):
                    var_path = match.group(1)
                    print(f"[DEBUG] Resolving variable path: {var_path}")
                    
                    # First try to resolve as a direct environment variable
                    if var_path in os.environ:
                        resolved = os.environ[var_path]
                        print(f"[DEBUG] Resolved from env var: {var_path} = {resolved}")
                        return resolved
                    
                    # Then try to resolve as a nested path in the config
                    parts = var_path.split('.')
                    current = root_config
                    
                    try:
                        # Try to resolve the full path first
                        full_path = var_path
                        for part in parts:
                            if isinstance(current, dict) and part in current:
                                current = current[part]
                                full_path += f".{part}"
                                print(f"[DEBUG] Resolved path part: {full_path} = {current}")
                            else:
                                # If we can't find the path, try to resolve each part
                                resolved_part = cls._resolve_part(part, root_config, full_path, root_config)
                                if resolved_part != part:
                                    current = resolved_part
                                    print(f"[DEBUG] Resolved partial path: {part} -> {resolved_part}")
                                else:
                                    raise KeyError(part)
                        
                        result = str(current)
                        print(f"[DEBUG] Successfully resolved full path: {var_path} = {result}")
                        return result
                        
                    except (KeyError, TypeError) as e:
                        print(f"[DEBUG] Could not resolve full path, trying part by part: {e}")
                        # If we can't resolve the full path, try to resolve each part
                        resolved_parts = []
                        for part in parts:
                            resolved = cls._resolve_part(part, root_config, f"{full_path}.{part}", root_config)
                            print(f"[DEBUG] Resolving part '{part}': {resolved}")
                            
                            if resolved == part and part not in os.environ:
                                print(f"[ERROR] Could not resolve part: {part}")
                                raise ValueError(
                                    f"Undefined variable path in config: {var_path} "
                                    f"(at {parent_key}.{key if parent_key else ''})"
                                )
                                
                            final_value = resolved if resolved != part else os.environ.get(part, part)
                            resolved_parts.append(str(final_value))
                            
                        result = '.'.join(resolved_parts)
                        print(f"[DEBUG] Resolved path part by part: {var_path} -> {result}")
                        return result
                
                try:
                    config_copy[key] = re.sub(r'\${([^}]+)}', replace_nested_var, value)
                except RecursionError:
                    raise ValueError(f"Circular reference detected in config at {parent_key}.{key}")
        
        return config_copy
    
    @classmethod
    def _get_nested_value(cls, d: Dict[str, Any], path: str):
        """
        Get a value from a nested dictionary using a dot notation path.
        
        Args:
            d: The dictionary to search in
            path: The dot notation path (e.g., 'paths.base_dir')
            
        Returns:
            The value if found, None otherwise
        """
        keys = path.split('.')
        current = d
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        return current

    @classmethod
    def _resolve_part(cls, part: str, config: Dict[str, Any], full_path: str, root_config: Optional[Dict[str, Any]] = None) -> Any:
        """
        Resolve a single part of a path against the config.
        
        Args:
            part: The part to resolve (e.g., 'paths' or 'base_dir')
            config: The current config level
            full_path: The full path being resolved (for error messages)
            
        Returns:
            The resolved value
            
        Raises:
            ValueError: If the part cannot be resolved
        """
        debug = True
        if debug:
            print(f"[DEBUG] Resolving part: '{part}' in path: '{full_path}'")
            print(f"[DEBUG] Available keys at this level: {list(config.keys())}")
        
        # 1. Try to resolve as a direct environment variable
        if part in os.environ:
            value = os.environ[part]
            if debug:
                print(f"[DEBUG] Found environment variable '{part}': {value}")
            return value
        
        # 2. Try direct lookup in current config
        if part in config:
            value = config[part]
            if debug:
                print(f"[DEBUG] Found direct key '{part}' in config")
            return str(value) if not isinstance(value, (dict, list)) else value
        
        # 3. If part contains dots, try to resolve as a nested path
        if '.' in part:
            # Always use root_config for nested path resolution
            root = root_config if root_config is not None else config
            nested_value = cls._get_nested_value(root, part)
            if nested_value is not None:
                if debug:
                    print(f"[DEBUG] Resolved nested path '{part}' from root: {nested_value}")
                return str(nested_value) if not isinstance(nested_value, (dict, list)) else nested_value
            # If direct resolution failed, try to resolve each part from root
            parts = part.split('.')
            resolved_parts = []
            for p in parts:
                resolved = cls._resolve_part(p, root, f"{full_path}.{p}", root)
                resolved_parts.append(str(resolved))
            result = '.'.join(resolved_parts)
            if debug:
                print(f"[DEBUG] Resolved path part by part from root: '{part}' -> '{result}'")
            return result
        
        # 4. Try to find the part in any level of the config
        found_value = cls._find_in_dict(config, part)
        if found_value is not None:
            if debug:
                print(f"[DEBUG] Found part '{part}' in nested config: {found_value}")
            return str(found_value) if not isinstance(found_value, (dict, list)) else found_value
        
        # 5. If we have a full path with dots, try to resolve it from the root
        if '.' in full_path:
            try:
                parts = full_path.split('.')
                current = config
                for p in parts:
                    if isinstance(current, dict) and p in current:
                        current = current[p]
                    else:
                        break
                else:
                    if debug:
                        print(f"[DEBUG] Resolved full path from context: '{full_path}': {current}")
                    return str(current) if not isinstance(current, (dict, list)) else current
            except (KeyError, TypeError) as e:
                if debug:
                    print(f"[DEBUG] Error resolving from context: {e}")
                pass
        
        # If we get here, we couldn't resolve the part
        if debug:
            print(f"[ERROR] Failed to resolve part: '{part}' in path: '{full_path}'")
            print(f"[ERROR] Available keys at this level: {list(config.keys())}")
        raise ValueError(f"Undefined variable path in config: {part} (at {full_path})")
    
    @classmethod
    def _find_in_dict(cls, d: Dict[str, Any], target_key: str):
        """
        Recursively search for a key in a nested dictionary.
        
        Args:
            d: The dictionary to search in
            target_key: The key to find
            
        Returns:
            The value if found, None otherwise
        """
        if not isinstance(d, dict):
            return None
            
        # Check current level
        if target_key in d:
            return d[target_key]
            
        # Recursively check nested dictionaries
        for key, value in d.items():
            if isinstance(value, dict):
                found = cls._find_in_dict(value, target_key)
                if found is not None:
                    return found
        
        return None

    @classmethod
    def load_config(cls, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML file and resolve all variables.
        
        Args:
            config_path: Path to the YAML configuration file
            
        Returns:
            Resolved configuration dictionary
        """
        print(f"[DEBUG] Loading config from: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("[DEBUG] Raw config loaded:")
        print(yaml.dump(config, default_flow_style=False))
        
        # First pass: resolve all variable references
        print("[DEBUG] Resolving environment variables...")
        resolved_config = cls.resolve_env_vars(config)
        
        print("[DEBUG] Resolved config:")
        print(yaml.dump(resolved_config, default_flow_style=False))
        
        return resolved_config
