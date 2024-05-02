import os
import sys
from src.mlops_project.logger import logging

def get_error_details(error, error_details):
    _, _, exc_tb = error_details
    filename = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Error occurred in Python script: {filename}, line number: {line_number}, error message: {str(error)}"

class CustomException(Exception):
    def __init__(self, error_message, error_details):
        super().__init__(error_message)
        self.error_details = get_error_details(error_message, error_details)

    def __str__(self):
        return self.error_details
