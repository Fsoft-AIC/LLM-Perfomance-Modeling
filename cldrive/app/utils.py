import json
import logging
import subprocess
from loguru import logger

def getOpenCLPlatforms(cldrive_exe) -> None:
    """
    Identify compatible OpenCL platforms for current system.
    """
    try:
        cmd = subprocess.Popen(
            "{} --clinfo".format(cldrive_exe).split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        stdout, stderr = cmd.communicate()
        if stderr:
            raise ValueError(stderr)
    except Exception as e:
        logger.error(cmd)
        logger.error(e)
    CL_PLATFORMS = list(
        platform for platform in stdout.split("\n") if len(platform) > 0
    )
    return CL_PLATFORMS

def get_kernel_info(cldrive_exe, kernel_path):
    cmd = f"{cldrive_exe} --srcs={kernel_path} --kernelinfo"
    proc = subprocess.Popen(
        cmd.split(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = proc.communicate()
    try:
        kernels = json.loads(stdout, strict=False)
        return kernels
    except Exception as e:
        raise Exception(f"""ERROR: Get kernel info failed with:\n
                        - output: {stdout}
                        - error: {stderr}
                        - exception: {e}""")

def setup_logging(logger_name: str = "gpu-code-gen") -> logging.Logger:
    """
    Set up logging for the project."""
    # Configure basic logging settings
    logging.basicConfig(level=logging.INFO)

    # Create a custom formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - Running function: %(module)s.%(funcName)s"
    )

    # Create handlers and set the formatter
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler("cldrive_runner.log")
    file_handler.setFormatter(formatter)

    # Create the logger and add handlers
    logger = logging.getLogger(logger_name)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger

# Description: This function is used to check the corredimension ctness of the generated kernel.
def detect_kernel_dimensions(kernel_code):
    dimensions = {
        "3D-global": ["get_global_id(2)", "get_global_size(2)", 
                      "get_group_id(2)", "get_num_groups(2)", "get_global_offset(2)"],
        "2D-global": ["get_global_id(1)", "get_global_size(1)", 
                      "get_group_id(1)", "get_num_groups(1)", "get_global_offset(1)"],
        "3D-local": ["get_local_id(2)", "get_local_size(2)"],
        "2D-local": ["get_local_id(1)", "get_local_size(1)"],
    }

    result = ""
    for dim, patterns in dimensions.items():
        if any(pattern in kernel_code for pattern in patterns):
            result += " " + dim
    if result == "":
        result = "1D"
    return result
