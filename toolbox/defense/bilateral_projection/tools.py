import numpy as np
import matplotlib.pyplot as plt

def bar(title, adv, x, y):

    print x
    plt.figure()
    plt.suptitle(title)
    plt.subplot(1, 2, 1)
    plt.bar(x, y)
    #plt.xticks(x, x)
    plt.xticks(('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
    plt.subplot(1, 2, 2)
    plt.imshow(adv)
    plt.show()


