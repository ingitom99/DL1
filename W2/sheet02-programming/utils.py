import numpy, numpy.random

def getdata():
    X  = numpy.array([[-1.086,  0.997,  0.283, -1.506]])
    T  = numpy.array([[-0.579]])
    return X,T

def getparams():
    W1 = numpy.array([[-0.339, -0.047,  0.746, -0.319, -0.222, -0.217],
                      [ 1.103,  1.093,  0.502,  0.193,  0.369,  0.745],
                      [-0.468,  0.588, -0.627, -0.319,  0.454, -0.714],
                      [-0.070, -0.431, -0.128, -1.399, -0.886, -0.350]])
    W2 = numpy.array([[ 0.379, -0.071,  0.001,  0.281, -0.359,  0.116],
                      [-0.329, -0.705, -0.160,  0.234,  0.138, -0.005],
                      [ 0.977,  0.169,  0.400,  0.914, -0.528, -0.424],
                      [ 0.712, -0.326,  0.012,  0.437,  0.364,  0.716],
                      [ 0.611,  0.437, -0.315,  0.325,  0.128, -0.541],
                      [ 0.579,  0.330,  0.019, -0.095, -0.489,  0.081]])
    W3 = numpy.array([[ 0.191, -0.339,  0.474, -0.448, -0.867,  0.424],
                      [-0.165, -0.051, -0.342, -0.656,  0.512, -0.281],
                      [ 0.678,  0.330, -0.128, -0.443, -0.299, -0.495],
                      [ 0.852,  0.067,  0.470, -0.517,  0.074,  0.481],
                      [-0.137,  0.421, -0.443, -0.557,  0.155, -0.155],
                      [ 0.262, -0.807,  0.291,  1.061, -0.010,  0.014]])
    W4 = numpy.array([[ 0.073],
                      [-0.760],
                      [ 0.174],
                      [-0.655],
                      [-0.175],
                      [ 0.507]])
    B1 = numpy.array([-0.760,  0.174, -0.655, -0.175,  0.507, -0.300])
    B2 = numpy.array([ 0.205,  0.413,  0.114, -0.560, -0.136,  0.800])
    B3 = numpy.array([-0.827, -0.113, -0.225,  0.049,  0.305,  0.657])
    B4 = numpy.array([-0.270])

    return [W1,W2,W3,W4],[B1,B2,B3,B4]
