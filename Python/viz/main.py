"""
`python python/viz/main.py --tag <tag>`

The --tag must match the tag PythonVisualizerBridge was constructed with on
the C++ side. NeuralImageRecreator passes the output filename's stem as the
tag, so e.g. if you're running `--o results/portrait.png`, run this as:

    python viz/main.py --tag portrait

Run with no --tag at all if you didn't change the C++ default ("INR_Default").
Safe to start before, during, or after the trainer it'll just wait/retry
until the shared memory exists.
"""
import argparse
import sys

from PySide6.QtWidgets import QApplication

from main_window import MainWindow


def main():
    parser = argparse.ArgumentParser(description="Live viewer for the INR training engine.")
    parser.add_argument("--tag", default="INR_Default",
                         help="Shared memory namespace, must match the C++ side's tag.")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = MainWindow(args.tag)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
