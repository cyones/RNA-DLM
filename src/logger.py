

class Logger():
    def __init__(self, path):
        self.file = open(path, 'w', buffering=1)

    def write(self, str):
        self.file.write(str)
        print(str, end="")

    def __del__(self):
        self.file.close()

