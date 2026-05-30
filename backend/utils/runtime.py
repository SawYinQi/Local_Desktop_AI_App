
import platform

# This module handles the selection of runtime backends based on host platform
def pick_backend() -> str:
    """
    Returns the name of the backend to use for the LLM and agents, based on the host platform.
    "openvino" for Intel CPUs and "transformers" for Apple Silicon and others.
    """
    system = platform.system()       # e.g. "Darwin", "Windows", "Linux" 
    machine = platform.machine()    # e.g. "x86_64", "AMD64", "arm64", "aarch64" 

    is_apple_silicon = (system == "Darwin" and machine in ("arm64", "aarch64"))
    is_intel = machine in ("x86_64", "AMD64")

    if is_apple_silicon:
        return "transformers"    # OpenVINO can't use Metal GPU; MPS is much better for Apple Silicon
    elif is_intel:
        return "openvino"        # OpenVINO optimized for Intel CPUs 
    else:
        return "transformers"    # default to transformers for unknown platforms 
