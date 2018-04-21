"""Tutorial on how to create a convolutional autoencoder* w/ Tensorflow.
https://github.com/pkmital/tensorflow_tutorials
Parag K. Mital, Jan 2016

*Modified from convolution autoencoder to a convolutional VAE using an MMD system.
MMD:
https://github.com/ShengjiaZhao/MMD-Variational-Autoencoder
Zhao, S. et al. (2017) INfoVAE: Information Maximizing Variational Autoencoders.
CoRR, abs/1706.0.
"""
import tensorflow as tf
import numpy as np
import math
from libs.activations import lrelu
from libs.utils import corrupt, montage_batch
import os

# %%
def autoencoder(input_shape=[None, 784],
                n_filters=[1, 10, 10, 10],
                filter_sizes=[3, 3, 3, 3],
                corruption=False,
                vae_hidden=10):
    """Build a deep denoising autoencoder w/ tied weights.

    Parameters
    ----------
    input_shape : list, optional
        Description
    n_filters : list, optional
        Description
    filter_sizes : list, optional
        Description

    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training

    Raises
    ------
    ValueError
        Description
    """
    # %%
    # input to the network
    x = tf.placeholder(
        tf.float32, input_shape, name='x')


    # %%
    # ensure 2-d is converted to square tensor.
    if len(x.get_shape()) == 2:
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(
            x, [-1, x_dim, x_dim, n_filters[0]])
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')
    current_input = x_tensor

    # %%
    # Optionally apply denoising autoencoder
    if corruption:
        current_input = corrupt(current_input)

    # %%
    # Build the encoder
    encoder = []
    shapes = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())
        W = tf.Variable(
            tf.random_uniform([
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                n_input, n_output],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = lrelu(
            tf.add(tf.nn.conv2d(
                current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output

    #fully connected layer to resize to VAE latent space
    last_size = current_input.get_shape().as_list()
    W_to_latent = tf.Variable(
            tf.random_uniform([
                last_size[1],#filter size equal to size of feature map (fully connected)
                last_size[2],
                n_output, vae_hidden], #from last output size into number of latent channels
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
    b_to_latent = tf.Variable(tf.zeros([vae_hidden]))

    #latent code
    z = tf.add(tf.nn.conv2d(
                current_input, W_to_latent, strides=[1, 1, 1, 1], padding='VALID'), b_to_latent)
    '''
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    '''
    encoder.reverse()
    shapes.reverse()

    #fully connected layer to resize from VAE latent space
    W_from_latent = tf.Variable(
            tf.random_uniform([
                last_size[1],
                last_size[2],
                n_output, vae_hidden], #from latent channels to last output size 
                                       #(order is reversed because conv2d_transpose will be used)
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
    b_from_latent = tf.Variable(tf.zeros([n_output]))

    #input for decoder
    current_input = tf.add(tf.nn.conv2d_transpose(
                z, W_from_latent,
                tf.stack([tf.shape(x)[0], last_size[1], last_size[2], last_size[3]]), 
                strides=[1, 1, 1, 1], padding='VALID'), b_from_latent)

    # %%
    # Build the decoder using the same weights
    for layer_i, shape in enumerate(shapes):
        n_input = current_input.get_shape().as_list()[3] 
        n_output = encoder[layer_i].get_shape().as_list()[2]#2 because reversing convolution
        W = tf.Variable(
            tf.random_uniform([
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                n_output, n_input],#reverse order because transposing
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros(n_output)) 
        output = lrelu(tf.add(
            tf.nn.conv2d_transpose(
                current_input, W,
                tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output

    # %%
    # now have the reconstruction through the network
    y = tf.nn.sigmoid(current_input)
    # cost function, prior MMD-loss and reconstruction loss
    z_dims = tf.shape(z)#z.get_shape().as_list()
    p_z = tf.random_normal(tf.stack(z_dims))#'true samples' from valid distribution
    loss_mmd = compute_mmd(p_z, z)
    loss_nll = tf.reduce_mean(tf.square(y - x_tensor))
    
    cost = loss_nll + loss_mmd

    # %%
    return {'x': x, 'z': z, 'y': y, 'cost': cost}


def compute_kernel(x_i, y_i):
    x_dims = tf.shape(x_i)    
    #squish height and width into z length to make it gaussian individually
    x = tf.reshape(x_i,[-1,x_dims[1]*x_dims[2]*x_dims[3]])
    y_dims = tf.shape(y_i)    
    y = tf.reshape(y_i,[-1,y_dims[1]*y_dims[2]*y_dims[3]])
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

def compute_mmd(x, y):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)

# %%
def test_mnist(result_dir='cVAE-mmd_Results',latent_dim=10):
    """Test the convolutional autoencder using MNIST."""
    # %%
    import tensorflow as tf
    import tensorflow.examples.tutorials.mnist.input_data as input_data
    import matplotlib.pyplot as plt

    #create results directory
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # %%
    # load MNIST as before
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    mean_img = np.mean(mnist.train.images, axis=0)
    ae = autoencoder(vae_hidden=latent_dim)

    # %%
    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    # %%
    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # %%
    # Fit all training data
    t_i = 0
    batch_size = 100
    n_epochs = 50
    n_examples = 20
    test_xs, _ = mnist.test.next_batch(n_examples)
    xs, ys = mnist.test.images, mnist.test.labels
    fig_manifold, ax_manifold = plt.subplots(1, 1)
    fig_reconstruction, axs_reconstruction = plt.subplots(2, n_examples, figsize=(10, 2))
    fig_image_manifold, ax_image_manifold = plt.subplots(1, 1)
    for epoch_i in range(n_epochs):
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, _ = mnist.train.next_batch(batch_size)
            train = np.array([img - mean_img for img in batch_xs])
            sess.run(optimizer, feed_dict={ae['x']: train})
        cur_cost,z_vect = sess.run([ae['cost'], ae['z']], feed_dict={ae['x']: train})
        print(epoch_i, cur_cost)
        
        # %%
        # Plot example reconstructions from latent layer
        imgs = []
        z_ref = np.zeros_like(z_vect[0:1,:]) 
        #use two random orthogonal directions
        z_1 = np.random.standard_normal(np.shape(z_ref[0,0,0,:])) # random vector
        z_1 /= np.linalg.norm(z_1)
        z_2 = np.random.standard_normal(np.shape(z_ref[0,0,0,:])) # random vector
        z_2 -= z_2.dot(z_1) * z_1 #keep only the component of z_2 that is orthogonal to z_1
        z_2 /= np.linalg.norm(z_2)
        
        for img_i in np.linspace(-3, 3, n_examples):
            for img_j in np.linspace(-3, 3, n_examples):
                z_in = z_ref + img_i*z_1 + img_j*z_2
                recon = sess.run(ae['y'], feed_dict={ae['z']: z_in,ae['x']: train[0:1,:]})
                imgs.append(np.reshape(recon, (1, 28, 28, 1)))
        imgs_cat = np.concatenate(imgs)
        ax_manifold.imshow(montage_batch(imgs_cat))
        fig_path = os.path.join(result_dir,'crosssection_of_manifold_%08d.png' % t_i)
        fig_manifold.savefig(fig_path)

        # %%
        # Plot example reconstructions
        recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs})
        print(recon.shape)
        for example_i in range(n_examples):
            axs_reconstruction[0][example_i].imshow(
                np.reshape(test_xs[example_i, :], (28, 28)),
                cmap='gray')
            axs_reconstruction[1][example_i].imshow(
                np.reshape(
                    np.reshape(recon[example_i, ...], (784,)),
                    (28, 28)),
                cmap='gray')
            axs_reconstruction[0][example_i].axis('off')
            axs_reconstruction[1][example_i].axis('off')
        fig_path = os.path.join(result_dir,'reconstruction_%08d.png' % t_i)
        fig_reconstruction.savefig(fig_path)

        
        # %%
        # Plot manifold of latent layer
        if latent_dim == 2:
            zs = sess.run(ae['z'], feed_dict={ae['x']: xs})
            ax_image_manifold.clear()
            ax_image_manifold.scatter(zs[:,0,0,0], zs[:,0,0,1],
                c=np.argmax(ys, 1), alpha=0.2)
            ax_image_manifold.set_xlim([-6, 6])
            ax_image_manifold.set_ylim([-6, 6])
            ax_image_manifold.axis('off')
            fig_path = os.path.join(result_dir,'image_manifold_%08d.png' % t_i)
            fig_image_manifold.savefig(fig_path)

        t_i += 1


# %%
if __name__ == '__main__':
    test_mnist()

