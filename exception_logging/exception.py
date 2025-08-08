import sys
import logging



def error_message_detail(error, error_detail):
    """Extracts the error message and details from an exception."""
    _, _, exc_tb = error_detail.exc_info()  # Extracts the traceback information about line number and file name
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "Unknown"
        line_number = "Unknown"
    error_message = f"Error occurred in Python script: [{file_name}] at line number: [{line_number}] error message: [{str(error)}]"
    return error_message



class CustomException(Exception):
    """Custom exception class to handle exceptions with detailed error messages."""
    
    def __init__(self, error_message, error_detail):
        """Initializes the CustomException with an error message."""
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        """Returns the string representation of the error message."""
        return self.error_message
    
if __name__ == "__main__":
    try:
        a = 1 / 0  # Example to raise an exception
    except Exception as e:
        logging.info("Divide by zero error occurred")
        raise CustomException(e, sys)