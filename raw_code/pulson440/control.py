#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Command and control script for PulsON 440 via Pi."""

__author__ = 'Ramamurthy Bhagavatula, Michael Riedl'
__version__ = '1.0'
__maintainer__ = 'Ramamurthy Bhagavatula'
__email__ = 'ramamurthy.bhagavatula@ll.mit.edu'

# Update path
from pathlib import Path
import sys
if Path('..//').resolve().as_posix() not in sys.path:
    sys.path.insert(0, Path('..//').resolve().as_posix())

# Import required modules and methods
import argparse
from common.helper_functions import is_valid_file, yes_or_no, deconflict_file, setup_logger, \
    close_logger
from pulson440.pulson440 import PulsON440
from pulson440.constants import DEFAULT_LOGGER_NAME, DEFAULT_LOGGER_CONFIG, FOREVER_SCAN_COUNT, \
    MIN_SCAN_COUNT, CONTINUOUS_SCAN_INTERVAL
import yaml

# Logger setup
try:
    logger_config_filename = (Path(__file__).parent / 'log_config.yml').resolve().as_posix()
    with open(logger_config_filename, 'r') as f:
        logger_config = yaml.load(f, Loader=yaml.FullLoader)
    logger = setup_logger(name=logger_config['name'], config=logger_config['config'])
except Exception as e:
    print(e)
    logger = setup_logger(name=DEFAULT_LOGGER_NAME, config=DEFAULT_LOGGER_CONFIG)

def parse_args(args):
    """Input argument parser.
    
    Args:
        args (list)
            Input arguments as taken from command line execution via sys.argv[1:].
    
    Returns:
        parsed_args (namespace)
            Parsed arguments.
            
    Raises:
        RuntimeError - too many arguments provided
        TODO: Update w/ appropriate error cases.
    """
    # List of arguments needed

    # settings_file

    # scan_data_filename

    # scan_count

    # return_data
    
    
    # Define argument parser
    # TODO: Insert argument parser; recommend usage of argparse library,
    # https://docs.python.org/3.5/library/argparse.html)
    arg_names = ["setting_files", "scan_count", "scan_data_filename", "return_data"] #Name of args
    args_def =  ['radar_settings.yml', FOREVER_SCAN_COUNT, None, None] #Default values of args

    # Creates a Namespace parsed_args
    parsed_args = argparse.Namespace()
    # Creates a dictionary object that parsed_args references
    args_dict = vars(parsed_args)

    #Checks to ensure at or bellow max numer of arguments
    if len(args) > len(arg_names):
        raise RuntimeError("Too many arguments provide. Expected: {0} Got: {1}".format(len(args_name), len(args)))

    # Creates dictionary entries matching arg value with arg names
    for a in range(len(args)):
        args_dict[arg_names[a]] = args[a]
    for d in range(len(arg_names) - len(args)):
        #Assign defalt values to unasigned args
        args_dict[arg_names[len(args)+d]] = args_def[arg_names[len(args)+d]]

    # Perform any needed additional checks and modifcation of parsed arguments
    # TODO: Insert appropriate code here
    if parsed_args is None:
        logger.info()
    
    return parsed_args

def main(args):
    """Main execution method to command radar to collect data.
    
    Args:
        args (list)
            Input arguments as taken from command line execution via sys.argv[1:].
    
    Returns:
        data (str)
            Data read from the radar; needs to unpacked to properly access scan information. Will 
            only be non-empty if return_data input flag is set to True.
    
    Raises:
        # TODO: Update w/ appropriate error cases.
    """
    logger.info('Starting radar data collection process...')

    # Parse input arguments
    parsed_args = parse_args(args)
    logger.debug('Input arguments are --> {0}'.format(parsed_args))

    # Initialize output
    data = None
    
    try:
        # Initialize radar
        radar = PulsON440(logger=logger)
        
        # TODO: Insert appropriate code that connects to radar, gets user settings, sets radar 
        # configuration, commands appropriate collection, and returns the collected data

        radar.connect()
        radar.read_setting_file(settings_file=parse_args.settings_file)
        radar.set_radar_config()
        if parsed_args.collect.mode == 'collect':
            data = radar.collect(scan_count=parsed_args.scan_count, scan_data_filename=parsed_args.scan_data_filename)
        elif parsed_args.collect.mode == 'quick':
            data = radar.quick_look(scan_data_filename=parsed_args.scan_data_filename)
        else:
            raise RuntimeError("Unrecognized collection mode {0}".format(parsed_args.collect_mode))
        logger.info('Completed radar data collection process!')

    except Exception:
        logger.exception('Fatal error encountered!')
        
    # Disconnect radar and close logger
    finally:
        radar.disconnect()
        close_logger(logger)
        
    return data
    
if __name__ == '__main__':
    """Standard Python alias for command line execution."""
    main(sys.argv[1:])
    
