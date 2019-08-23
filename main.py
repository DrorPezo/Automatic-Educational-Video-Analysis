import os
import sys


def main():
    path = os.path.join('Gui', 'gui.py')
    os.system('python ' + path)


if __name__ == "__main__":
    sys.path.append('utils')
    main()
