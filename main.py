import argparse
import os
import sys
from humor_model import evaluate_task1, evaluate_task2, evaluate_task3

def parse_model_args(cls_args):
    parser = argparse.ArgumentParser(description="")
    """ command line configs """
    parser.add_argument("-task", type=str, default="task1a", choices=["task1a", "task1b","task2a"])

    args = parser.parse_args(cls_args)
    print("Args: ", args.task)
    return args


args = parse_model_args(sys.argv[1:]) 

def main(task):
    if task== "task1b":
        return evaluate_task2()
    elif task == 'task2a':
        return evaluate_task3()
    return evaluate_task1()

if __name__ == "__main__":
    
    main(args.task)
