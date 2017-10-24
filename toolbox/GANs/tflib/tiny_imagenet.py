import numpy as np
import scipy.misc
import time
import glob

data_dir = '/home/neale/repos/adversarial-toolbox/images/imagenet10'
label_dir = '/home/neale/repos/adversarial-toolbox/images/imagenet10'


def make_generator(path, n_files, batch_size):
    epoch_count = [1]
    filelist = glob.glob(path+'/*.JPEG')

    def get_epoch():
        images = np.zeros((batch_size, 3, 64, 64), dtype='int32')
        labels = np.empty((batch_size), dtype='int32')
        files = range(n_files)
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        random_state.shuffle(filelist)
        epoch_count[0] += 1
        for n, i in enumerate(files):
            f = filelist[i]
            lx = f[-(f[::-1].index('/')):]  # please dont hurt me
            label = int(lx[0]) # I just need to isolate the label

            image = scipy.misc.imread(f, mode='RGB')
            image = scipy.misc.imresize(image, (64, 64, 3))
            labels[n % batch_size] = label
            images[n % batch_size] = image.transpose(2, 0, 1)
            if n > 0 and n % batch_size == 0:
                yield (images,labels)
    return get_epoch


def load(batch_size, data_dir=data_dir, label_dir=label_dir):
    return (
        make_generator(data_dir+'/train', 4999, batch_size),
        make_generator(data_dir+'/val', 499, batch_size)
    )


if __name__ == '__main__':
    train_gen, valid_gen = load(64)
    t0 = time.time()
    for i, batch in enumerate(train_gen(), start=1):
        print "{}\t{}".format(str(time.time() - t0), batch[0][0, 0, 0, 0])
        if i == 1000:
            break
        t0 = time.time()
