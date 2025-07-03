import sys
import traceback

def error_message_detail(error, error_detail: sys):
    """
    error: The actual error object
    error_detail: pass `sys` so we can get traceback
    """
    _, _, exc_tb = sys.exc_info()

    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = f"Error occurred in python script name [{file_name}] line number [{line_number}] error_message [{str(error)}]"
    else:
        # Safe fallback if traceback is not available
        error_message = f"Error: {str(error)}"

    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
