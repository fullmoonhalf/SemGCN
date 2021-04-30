import argparse




def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')
    args = parser.parse_args()
    return args

def main(args):
    print("Entry Point.")
    return

if __name__ == '__main__':
    main(parse_args())
