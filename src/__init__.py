"""
=============================================================================
SRC PACKAGE INITIALIZER
=============================================================================

This file makes the 'src' folder a Python package.
The 'src' folder contains main application scripts:

- collect_data.py: Data collection application
- recognize.py: Real-time recognition application (Phase 3)

These are the scripts users will run directly.

HOW TO RUN:
-----------
    # Data collection
    python -m src.collect_data
    
    # Recognition (after training)
    python -m src.recognize

WHY 'python -m'?
----------------
Using 'python -m src.collect_data' (module mode) instead of 
'python src/collect_data.py' (file mode) ensures:
- Proper package imports work correctly
- Python path is set up properly
- Relative imports work as expected

=============================================================================
"""

# ============================================================================
# PACKAGE INFO
# ============================================================================
__version__ = "1.0.0"
__author__ = "Sign Language App Team"

# ============================================================================
# AVAILABLE SCRIPTS
# ============================================================================
# List of main scripts in this package
AVAILABLE_SCRIPTS = [
    "collect_data",    # Data collection tool
    # "recognize",     # Will be added in Phase 3
]