"""
UrsaML Parser - Parses and serializes Ursa Markup Language files
"""
import re
from typing import Dict, List, Tuple, Any
import json

def parse_ursaml(content: str) -> Dict[str, Any]:
    """
    Parse UrsaML content into a structured format.
    
    Returns:
        {
            'version': str,
            'identifier': str,
            'columns': List[str],
            'column_values': Dict[str, Any],
            'structure': List[Tuple[str, str, float, str]],  # (source, target, weight, type)
            'nodes': Dict[str, Dict]  # node_id -> {columns: {...}, detailed: {...}}
        }
    """
    result = {
        'version': '0.1',
        'identifier': '',
        'columns': [],
        'column_values': {},
        'structure': [],
        'nodes': {}
    }
    
    lines = content.strip().split('\n')
    current_section = None
    section_lines = {'COLUMNS': [], 'STRUCTURE': [], 'CONTENT': []}
    
    # Parse header
    for i, line in enumerate(lines):
        line = line.strip()
        
        if line.startswith('@URSAML'):
            parts = line.split()
            if len(parts) >= 2:
                result['version'] = parts[1]
            if len(parts) >= 3:
                result['identifier'] = parts[2]
        
        elif line == '@COLUMNS':
            current_section = 'COLUMNS'
        elif line == '@STRUCTURE':
            current_section = 'STRUCTURE'
        elif line == '@CONTENT':
            current_section = 'CONTENT'
        elif line in ['@ENDCOLUMNS', '@ENDSTRUCTURE', '@ENDCONTENT']:
            current_section = None
        elif line.startswith('@END '):
            break
        elif current_section and not line.startswith('@'):
            if line:  # Skip empty lines
                section_lines[current_section].append(line)
    
    # Parse columns
    for line in section_lines['COLUMNS']:
        if ':' in line:
            col_name, value_str = line.split(':', 1)
            result['columns'].append(col_name)
            # Parse the value (could be array or single value)
            try:
                result['column_values'][col_name] = json.loads(value_str)
            except:
                result['column_values'][col_name] = value_str
    
    # Parse structure (edges)
    for line in section_lines['STRUCTURE']:
        if '->' in line:
            parts = line.split('->')
            source = parts[0].strip()
            rest = parts[1].split(':')
            target = rest[0].strip()
            weight = float(rest[1]) if len(rest) > 1 else 1.0
            edge_type = rest[2].strip('"') if len(rest) > 2 else 'default'
            result['structure'].append((source, target, weight, edge_type))
    
    # Parse content (nodes)
    current_node = None
    node_content = []
    
    for line in section_lines['CONTENT']:
        # Check if this is a new node definition
        if '|' in line and not line.startswith(' '):
            # Process previous node if exists
            if current_node:
                result['nodes'][current_node['id']] = {
                    'columns': current_node['columns'],
                    'detailed': parse_detailed_content('\n'.join(node_content))
                }
            
            # Start new node
            parts = line.split('|')
            node_id = parts[0].strip()
            
            # Extract column values
            column_values = {}
            for i, col_name in enumerate(result['columns']):
                if i + 1 < len(parts):
                    value = parts[i + 1].strip().strip('"')
                    try:
                        # Try to convert to appropriate type
                        if '.' in value:
                            column_values[col_name] = float(value)
                        else:
                            column_values[col_name] = value
                    except:
                        column_values[col_name] = value
            
            current_node = {'id': node_id, 'columns': column_values}
            node_content = []
            
            # Check if detailed content starts on same line
            if len(parts) > len(result['columns']) + 1:
                remaining = parts[-1]
                if remaining.strip().startswith('{'):
                    node_content.append(remaining)
        else:
            # Continuation of node content
            node_content.append(line)
    
    # Don't forget the last node
    if current_node:
        result['nodes'][current_node['id']] = {
            'columns': current_node['columns'],
            'detailed': parse_detailed_content('\n'.join(node_content))
        }
    
    return result

def parse_detailed_content(content: str) -> Dict[str, Any]:
    """Parse the detailed content section of a node."""
    content = content.strip()
    if not content or content == '{}':
        return {}
    
    # Remove outer braces
    if content.startswith('{') and content.endswith('}'):
        content = content[1:-1]
    
    result = {'params': {}, 'meta': {}}
    
    # Parse key:value pairs
    lines = content.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        if ':' in line:
            # Handle lines like "param:framework:tensorflow"
            if line.startswith('param:'):
                # Remove 'param:' prefix
                rest = line[6:]
                if ':' in rest:
                    key, value = rest.split(':', 1)
                    value = value.strip().strip('"')
                    # Convert types
                    if value.isdigit():
                        value = int(value)
                    elif value.replace('.', '').isdigit():
                        value = float(value)
                    elif value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                    result['params'][key] = value
            elif line.startswith('meta:'):
                # Remove 'meta:' prefix
                rest = line[5:]
                if ':' in rest:
                    key, value = rest.split(':', 1)
                    value = value.strip().strip('"')
                    # Convert types
                    if value.isdigit():
                        value = int(value)
                    elif value.replace('.', '').isdigit():
                        value = float(value)
                    elif value.lower() in ['true', 'false']:
                        value = value.lower() == 'true'
                    result['meta'][key] = value
            else:
                # Regular key:value
                key, value = line.split(':', 1)
                value = value.strip().strip('"')
                # Convert types
                if value.isdigit():
                    value = int(value)
                elif value.replace('.', '').isdigit():
                    value = float(value)
                elif value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                result[key] = value
    
    return result

def serialize_ursaml(data: Dict[str, Any]) -> str:
    """
    Serialize a graph structure back to UrsaML format.
    """
    lines = []
    
    # Header
    version = data.get('version', '0.1')
    identifier = data.get('identifier', 'untitled')
    lines.append(f"@URSAML {version} {identifier}")
    lines.append("")
    
    # Columns section
    lines.append("@COLUMNS")
    for col_name in data.get('columns', []):
        values = data.get('column_values', {}).get(col_name, [])
        if isinstance(values, list):
            values_str = json.dumps(values)
        else:
            values_str = str(values)
        lines.append(f"{col_name}:{values_str}")
    lines.append("@ENDCOLUMNS")
    lines.append("")
    
    # Structure section
    lines.append("@STRUCTURE")
    for edge in data.get('structure', []):
        source, target, weight, edge_type = edge
        lines.append(f"{source}->{target}:{weight}:\"{edge_type}\"")
    lines.append("@ENDSTRUCTURE")
    lines.append("")
    
    # Content section
    lines.append("@CONTENT")
    for node_id, node_data in data.get('nodes', {}).items():
        # Build the pipe-separated line
        parts = [node_id]
        
        # Add column values
        columns = node_data.get('columns', {})
        for col_name in data.get('columns', []):
            value = columns.get(col_name, '')
            if isinstance(value, str):
                parts.append(f'"{value}"')
            else:
                parts.append(str(value))
        
        # Start detailed content
        parts.append('{')
        lines.append('|'.join(parts))
        
        # Add detailed content
        detailed = node_data.get('detailed', {})
        
        # Add direct properties
        for key, value in detailed.items():
            if key not in ['params', 'meta']:
                if isinstance(value, str):
                    lines.append(f"    {key}:\"{value}\"")
                else:
                    lines.append(f"    {key}:{value}")
        
        # Add params
        for key, value in detailed.get('params', {}).items():
            if isinstance(value, str):
                lines.append(f"    param:{key}:\"{value}\"")
            else:
                lines.append(f"    param:{key}:{value}")
        
        # Add meta
        for key, value in detailed.get('meta', {}).items():
            if isinstance(value, str):
                lines.append(f"    meta:{key}:\"{value}\"")
            else:
                lines.append(f"    meta:{key}:{value}")
        
        lines.append("}")
        lines.append("")
    
    lines.append("@ENDCONTENT")
    lines.append("")
    lines.append(f"@END {identifier}")
    
    return '\n'.join(lines) 