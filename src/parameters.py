import torch as tr
import getopt
import math
import sys

help_message = "model_fit.py -i <input_fastas> -m <model>"

class ParameterParser():
    def __init__(self, argv):
        self.input_files = []
        self.model_file = 'model.pmt'
        self.device = tr.device('cpu')
        try:
            opts, args = getopt.getopt(argv, "hi:m:",
                    ["input_fasta=", "model="])
        except getopt.GetoptError:
            print(help_message)
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print(help_message)
                sys.exit()
            elif opt in ("-i", "--input_fasta"):
                self.input_files.append(arg)
            elif opt in ("-m", "--model"):
                self.model_file = arg

