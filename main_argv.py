import sys
import argparse
def main(args):
    arg_names = ["a","b","c"]

    #Creates a Namespace parsed_args
    parsed_args = argparse.Namespace()

    #Creates a dictionary object that parsed_args references
    args_dict = vars(parsed_args)

    #Creates dictionary entries matching arg value with arg names
    for a in range(len(args)):
        args_dict[arg_names[a]] = args[a]

    return parsed_args

if __name__ == '__main__':
    """Standard Python alias for command line execution."""
    main(sys.argv[1:])