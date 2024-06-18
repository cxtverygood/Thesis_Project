import pickle
import bz2


def load_file(fname):
    if fname.endswith(".pbz2"):
        with bz2.BZ2File(fname, 'r') as h:
            data = pickle.load(h)
    else:
        with open(fname, "rb") as h:
            data = pickle.load(h)

    return data