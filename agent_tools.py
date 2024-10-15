import io
import contextlib
import re
import sympy as sp
import traceback

from RestrictedPython import safe_builtins, utility_builtins
from langchain_core.tools import tool

other_whitelisted_builtins = {
    "min": min,
    "max": max,
    "map": map,
    "sum": sum,
    "type": type,
    "next": next,
    "filter": filter,
    "all": all,
    "enumerate": enumerate,
    "dict": dict,
    "any": any,
    "bin": bin,
    "range": range,
    "list": list,
    "tuple": tuple,
    "reversed": reversed,
    "iter": iter,
    "object": object,
    "print": print,
}

# Global dictionary to store variables
persistent_globals = {
    '__builtins__': {
        **safe_builtins,
        **utility_builtins,
        **other_whitelisted_builtins,
    }
}

@tool
def sympy_execute(code: str):
    """
    Call this tool to perform symbolic math.
    Input should be valid Python code that uses sympy.
    All sympy tools are imported by default so you dont need to import them again.
    Other imports are not allowed.
    """
    # Prepend the SymPy import statement
    # preamble = "import sympy as sp\n"
    preamble = "from sympy import *\nimport sympy as sp\n"
    
    commands = code.split(';')
    commands = [cmd.strip() for cmd in commands if cmd.strip()]
    code = '\n'.join(commands)

    # Remove any import statements from the AI's code
    sanitized_code = re.sub(r'^\s*import\s+[^\n]+', '', code, flags=re.MULTILINE)
    sanitized_code = re.sub(r'^\s*from\s+[^\n]+', '', sanitized_code, flags=re.MULTILINE)
    
    # Combine the preamble with the sanitized code
    final_code = preamble + sanitized_code
    # print(f'executing code: \n{final_code}')
    
    _SAFE_MODULES = frozenset(["sympy"])
    def _safe_import(name, *args, **kwargs):
        if name not in _SAFE_MODULES:
            raise Exception(f"module not in whitelist: {name!r}")
        return __import__(name, *args, **kwargs)

    # Add the safe import function to the persistent globals
    persistent_globals['__builtins__']['__import__'] = _safe_import
    
    # Create a string buffer to capture the output
    output_buffer = io.StringIO()
    result = None
    
    try:
        # Redirect stdout to the string buffer
        with contextlib.redirect_stdout(output_buffer):
            print(f"Executing code:\n{sanitized_code}")
            # Execute the final code in the persistent environment
            exec(final_code, persistent_globals)
            # Capture the result of the last expression if it's an expression
            last_line = sanitized_code.split('\n')[-1]
            try:
                compiled_code = compile(last_line, '<string>', 'eval')
                result = eval(compiled_code, persistent_globals)
            except SyntaxError:
                pass
    except Exception as e:
        # Capture the traceback
        tb = traceback.format_exc()
        # Print the error message and traceback
        print(tb)
        error_msg = f"Error: {e}"
        print(error_msg)
        return error_msg
    
    # Get the output from the string buffer
    output = output_buffer.getvalue()
    if result is not None:
        output += f"\nResult: {result}"
    return output

def reset_persistent_globals():
    """
    Reset the persistent globals to their initial state.
    """
    global persistent_globals
    persistent_globals = {
        '__builtins__': {
            **safe_builtins,
            **utility_builtins,
            **other_whitelisted_builtins,
        }
    }

def create_tools():
    return [sympy_execute]


