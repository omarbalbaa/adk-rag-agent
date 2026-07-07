"""
Tool for exporting data to an Excel file.
"""

import logging
import os
from typing import List, Dict, Any
import base64
import pandas as pd
from google.adk.tools.tool_context import ToolContext

try:
    from shared_state import latest_generated_files
except ImportError:
    latest_generated_files = None

def export_excel(
    data: List[Dict[str, Any]],
    filename: str,
    tool_context: ToolContext,
) -> dict:
    """
    Export a list of dictionaries (records) to an Excel file.

    Args:
        data (List[Dict[str, Any]]): The data to export, where each dictionary represents a row.
        filename (str): The name or path of the output Excel file. E.g., 'results.xlsx'.
        tool_context (ToolContext): The tool context.

    Returns:
        dict: The result of the export operation.
    """
    try:
        if not data:
            return {
                "status": "warning",
                "message": "No data provided to export.",
                "filename": filename
            }

        if not filename.endswith('.xlsx'):
            filename += '.xlsx'

        # Get user ID from tool context
        user_id = tool_context._invocation_context.user_id
        
        # Create output directory
        output_dir = os.path.join(os.getcwd(), 'generated-sheets', user_id)
        os.makedirs(output_dir, exist_ok=True)
        
        file_path = os.path.join(output_dir, filename)

        df = pd.DataFrame(data)
        
        df.to_excel(file_path, index=False)

        # Read the file and encode to base64
        with open(file_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode('utf-8')
            
        # Delete the file to save storage space
        try:
            os.remove(file_path)
        except OSError as e:
            logging.warning(f"Failed to delete temporary file {file_path}: {e}")
        
        # Add to shared state
        if latest_generated_files is not None:
            current_files = latest_generated_files.get()
            new_files = current_files.copy()
            new_files.append({
                "name": os.path.basename(file_path),
                "data": encoded_string,
                "type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            })
            latest_generated_files.set(new_files)

        return {
            "status": "success",
            "message": f"Successfully exported data to {file_path} and added to response files.",
            "filename": file_path,
            "rows_exported": len(data),
            "file_base64": encoded_string
        }

    except ImportError:
        error_msg = "pandas or openpyxl is not installed. Please install them to use this tool."
        logging.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "filename": filename
        }
    except Exception as e:
        error_msg = f"Error exporting to Excel: {str(e)}"
        logging.error(error_msg)
        return {
            "status": "error",
            "message": error_msg,
            "filename": filename
        }
