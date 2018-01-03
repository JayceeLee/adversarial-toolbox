import abc
import sys
import warnings
import numpy as np

 """ Base class for adversarial attacks
    -----------------------------------------
    An attack may be implemented in different frameworks.
    This class can sit on top of libraries like cleverhans and foolbox
    Those frameworks provide an easy way to wrap networks and attack images

    This class will return a set of images, given just a network name,
    making development faster
    """
class Attack(ABC):

    """ An attack takes a model and a behavior, the model is just a name,
    like "resnet" or "inceptionresnetv2". The behvaior is how the adversarial
    examples relate to the input data. This describes targeted vs untareted
    attacks, and by how much the adversarial example is misclassified
    Most behaviors are implemented as foolbox criteria
    """

    def __init__(self, model, behavior):
        self.__default_model = model
        self.__default_behavior = behavior

    """ Applies an attack to a specified image
    Assumes an initialized attack with network and behavior
    the label is not needed usually, and mostly applies to virtual adversarial
    examples.
    * Eventually a dataset option will be supported, but currently imagenet
    is used
    """
    def __call__(self, image, label=None, dataset=None, **kwargs):

        if image is None:
            warnings.warn("Image was of type 'None', returning None",
                          RuntimeWarning)

        if isinstance(image, Adversarial):
            if label is not None:
                raise ValueError('Label must not be passed when image is an
				 Adversarial instance')
            else:
                find = image
        else:
            if label is None:
                raise ValueError('Label must be passed when image is not an
				 Adversarial instance')
            else:
                model = self.__default_model
                behavior = self.__default_behavior
                if model is None or behavior is None:
                    raise ValueError('The attack needs to be initialized with
				     a model and a behavior or it needs to be
				     called with an adversarial attack.')
                find = Adversarial(model, criterion, image, label)

        assert find is not None

        adversarial = find

        if adversarial.distance.value == 0.:
            warnings.warn('Not running the attack because the original image is
			  already misclassified and the adversarial thus has a
			  distance of 0.')
        else:
            _ = self._apply(adversarial, **kwargs)
            assert _ is None, '_apply must return None'

        if adversarial.image is None:
            warnings.warn('{} did not find an adversarial, maybe the model or
 			  the criterion is not supported by this
			  attack.'.format(self.name()))

        if unpack:
            return adversarial.image
        else:
            return adversarial



