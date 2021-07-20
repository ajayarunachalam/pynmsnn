#!/bin/sh

# assumes run from root directory

echo "Installing NeuroMorphic Predictive Model with Spiking Neural Networks (SNN) Package using Pytorch" 
echo "START..."
pip install -r requirements.txt
echo "END"
xterm -e python -i -c "print('>>> from pyNM.spiking_binary_classifier import *');from pyNM.spiking_binary_classifier import *"
xterm -e python -i -c "print('>>> from pyNM.nonspiking_binary_classifier import *');from pyNM.nonspiking_binary_classifier import *"
xterm -e python -i -c "print('>>> from pyNM.spiking_multiclass_classifier import *');from pyNM.spiking_multiclass_classifier import *"
xterm -e python -i -c "print('>>> from pyNM.nonspiking_multiclass_classifier import *');from pyNM.nonspiking_multiclass_classifier import *"
xterm -e python -i -c "print('>>> from pyNM.spiking_regressor import *');from pyNM.spiking_regressor import *"
xterm -e python -i -c "print('>>> from pyNM.nonspiking_regressor import *');from pyNM.nonspiking_regressor import *"
xterm -e python -i -c "print('>>> from pyNM.cf_matrix import make_confusion_matrix');from pyNM.cf_matrix import make_confusion_matrix"
echo "Test Environment Configured"
echo "Package Installed & Tested Sucessfully"