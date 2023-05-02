"""
Module containing paths to the data, logs, scripts and checkpoints.

"""
import os

CHECKPOINTS_DIR = os.environ.get('CHECKPOINTS_DIR', '../checkpoints/')
DATA_DIR = os.environ.get('DATA_DIR', '../data/')
NOTEBOOKS_DIR = os.environ.get('NOTEBOOKS_DIR', '../notebooks/')
LOGS_DIR = os.environ.get('LOGS_DIR', '../logs/')
SCRIPTS_DIR = os.environ.get('SCRIPTS_DIR', '../scripts/')
STRUCTURES_DIR = os.environ.get('STRUCTURES_DIR', '/Users/kevinmicha/Downloads/all_structures/chothia/')