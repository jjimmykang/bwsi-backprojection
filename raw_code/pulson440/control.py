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
from unpack import unpack
from datetime import datetime as dt

# Logger setup
try:
    logger_config_filename = (Path(__file__).parent / 'log_config.yml').resolve().as_posix()
    with open(logger_config_filename, 'r') as f:
        logger_config = yaml.load(f, Loader=yaml.FullLoader)
    logger = setup_logger(name=logger_config['name'], config=logger_config['config'])
except Exception as e:
    print(e)
    logger = setup_logger(name=DEFAULT_LOGGER_NAME, config=DEFAULT_LOGGER_CONFIG)

def parse_args():
    '''Parses command line arguments.
    Returns:
        parsed_args(object)
            command line arguments parsed
            attributes:
                scan_mode(str): collect or quicklook
                scan_count(int): number of scans
                --settings: force a settings file path
                --create: automatically creates data file
                --noreturn: forces no return data
    '''

    # Set up parser for command line arguments
    parser = argparse.ArgumentParser(description='Pulson440 Radar software')
    parser.add_argument('scan_mode', type=str, help='collect or quicklook')
    parser.add_argument('scan_count', type=int, help='Scancount.')
    parser.add_argument('--settings', type=str, help='force a settings file path ')
    parser.add_argument('--nocreate', help='automatically create data file')
    parser.add_argument('--noreturn', help='forces no return data', action='store_true')

    # Object to hold command line arguments
    args = parser.parse_args()

    return args

def main():
    """Main execution method to command radar to collect data.
    Returns:
        data (str)
            Data read from the radar; needs to unpacked to properly access scan information. Will
            only be non-empty if return_data input flag is set to True.
    Raises:
        # TODO: Update w/ appropriate error cases.
    """
    logger.info('Starting radar data collection process...')


    # Fetch/Parse input arguments
    args = parse_args()
    logger.debug('Input arguments are --> {0}'.format(args))

    # Initialize output
    data = None

    try:
        # Initialize radar
        radar = PulsON440(logger=logger)

        # TODO: Insert appropriate code that connects to radar, gets user settings, sets radar
        # configuration, commands appropriate collection, and returns the collected data

        radar.connect()

        # set up settings file
        if args.settings:
            radar.read_settings_file(settings_file=args.settings)
        else:
            radar.read_settings_file(settings_file='./radar_settings.yml')

        # set up data file - mark with data if autocreate
        data_file = ''
        if args.nocreate:
            data_file = args.nocreate
        else:
            # name data_file with timestamp
            data_file = 'collected_data/scan_{0}{1}{2}{3}.txt'.format(dt.now().day, dt.now().hour, dt.now().minute, dt.now().second)
            # create the file
            f = open(data_file, 'w+')
            logger.info('opened new data file')

        # set up return data
        return_data_flag = False
        if args.noreturn:
            return_data_flag = True

        radar.set_radar_config()

        # Checks whether to collect or quickscan
        if args.scan_mode == 'collect':
            data = radar.collect(scan_count=args.scan_count, scan_data_filename=data_file, return_data=return_data_flag)
        elif args.scan_mode == 'quick':
            data = radar.quick_look(scan_data_filename=data_file, return_data=return_data_flag)
        elif args.scan_mode == 'noscan':
            logger.info('noscan detected... unpacking data.')
        else:
            raise RuntimeError('Unrecognized collection mode {0}'.format(parsed_args.collect_mode))
        logger.info('Completed radar data collection process!')

    except Exception:
        logger.exception('Fatal error encountered!')

    #try to unpack the data
    try:
        logger.info('Attempting to unpack data...')
        data_unpacked = unpack(data_file)

        # Simply prints the unpacked data
        print("data_unpacked:")
        print(data_unpacked)

    except Exception:
        logger.exception('Fatal error encountered!')

    # Disconnect radar and close logger
    finally:
        radar.disconnect()
        close_logger(logger)

    return data

if __name__ == '__main__':
    """Standard Python alias for command line execution."""
    main()
