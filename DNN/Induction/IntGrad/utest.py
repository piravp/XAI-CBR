import unittest
import numpy as np

from IntegratedGradients import *

from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers.core import Activation

##########################################
# Potentially add tests for other axioms #
#  - null effects                        #
#  - completeness (implemented)          #
#  - linearity                           #
#  - Symmetry                            #
##########################################

# python -m unittest DNN/Induction/IntGrad/utest.py
class test(unittest.TestCase):

    def test_completeness(self):
        X = np.array([[float(j) for j in i.rstrip().split(",")[:-1]] for i in open("DNN/Induction/IntGrad/notebooks/iris.data").readlines()][:-1])
        Y = np.array([0 for i in range(100)] + [1 for i in range(50)])

        model = Sequential([
            Dense(1, input_dim=4),
            Activation('sigmoid'),
        ])
        model.compile(optimizer='sgd', loss='binary_crossentropy')
        
        model.fit(X, Y,
          epochs=300, batch_size=10,
          validation_split=0.1, verbose=0)
        total = model.predict(X[0:1, :])[0,0]-model.predict(np.zeros((1,4)))[0,0]
        
        ig = integrated_gradients(model, verbose=0)
        explanation = ig.explain(X[0], num_steps=100000)
        print(total, explanation)
        self.assertAlmostEqual(total, np.sum(explanation), places=3)

if __name__ == '__main__':
    unittest.main()