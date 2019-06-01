import numpy as np
from io import BytesIO
from subprocess import Popen, PIPE
import sys

#@profile
def to_xnet(g, file_name, names=True):

    """
    Convert igraph object to a .xnet format string. This string
    can then be written to a file or used in a pipe. The purpose
    of this function is to have a fast means to convert a graph
    without any attributes to the .xnet format.

    Parameters
    ----------
    g : igraph.Graph
        Input graph.

    Returns
    -------
    strOut : string
        String that can be written to a .xnet file.
    """
    with open(file_name, 'w') as xnet_file:
        N = g.vcount()
        xnet_file.write('#vertices '+str(N)+' nonweighted\n')
        if names:
            for v in g.vs['name']:
                xnet_file.write('"' + v + '"' + '\n')
        xnet_file.write('#edges weighted undirected\n')

        for e, w in zip(g.es, g.es['weight']):
            xnet_file.write('%d %d %f\n' % (e.source, e.target, w))


def extract_motif():
    pass


def extract_weighted_motif():
    pass
