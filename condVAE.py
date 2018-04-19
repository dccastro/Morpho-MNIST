"""Training a variational autoencoder with 2 layer fully-connected
encoder/decoders and gaussian noise distribution.
https://github.com/pkmital/tensorflow_tutorials
Parag K. Mital, Jan 2016
"""
import tensorflow as tf
import numpy as np
from libs.utils import weight_variable, bias_variable, montage_batch
import os


# %%
def VAE(input_shape=[None, 784],
        n_components_encoder=2048,
        n_components_decoder=2048,
        n_hidden=2,
        debug=False, cond_shape=[None, 10]):
    # %%
    # Input placeholder
    if debug:
        input_shape = [50, 784]
        x = tf.Variable(np.zeros((input_shape), dtype=np.float32))
        cond_shape = [50, 10]
        cond = tf.Variable(np.zeros((cond_shape), dtype=np.float32))
    else:
        x = tf.placeholder(tf.float32, input_shape)
        cond = tf.placeholder(tf.float32, cond_shape)

    activation = tf.nn.softplus
    
    x_cond = tf.concat([x,cond],1)#combine conditional information
    dims = x.get_shape().as_list()
    n_features = dims[1]
    n_features_c = x_cond.get_shape().as_list()[1]

    W_enc1 = weight_variable([n_features_c, n_components_encoder])
    b_enc1 = bias_variable([n_components_encoder])
    h_enc1 = activation(tf.matmul(x_cond, W_enc1) + b_enc1)

    W_enc2 = weight_variable([n_components_encoder, n_components_encoder])
    b_enc2 = bias_variable([n_components_encoder])
    h_enc2 = activation(tf.matmul(h_enc1, W_enc2) + b_enc2)

    W_enc3 = weight_variable([n_components_encoder, n_components_encoder])
    b_enc3 = bias_variable([n_components_encoder])
    h_enc3 = activation(tf.matmul(h_enc2, W_enc3) + b_enc3)

    W_mu = weight_variable([n_components_encoder, n_hidden])
    b_mu = bias_variable([n_hidden])

    W_log_sigma = weight_variable([n_components_encoder, n_hidden])
    b_log_sigma = bias_variable([n_hidden])

    z_mu = tf.matmul(h_enc3, W_mu) + b_mu
    z_log_sigma = 0.5 * (tf.matmul(h_enc3, W_log_sigma) + b_log_sigma)

    # %%
    # Sample from noise distribution p(eps) ~ N(0, 1)
    if debug:
        epsilon = tf.random_normal(
            [dims[0], n_hidden])
    else:
        epsilon = tf.random_normal(
            tf.stack([tf.shape(x)[0], n_hidden]))

    # Sample from posterior
    z = z_mu + tf.exp(z_log_sigma) * epsilon
    z_cond = tf.concat([z,cond],1)#combine conditional information
    n_hidden_c = z_cond.get_shape().as_list()[1]

    W_dec1 = weight_variable([n_hidden_c, n_components_decoder])
    b_dec1 = bias_variable([n_components_decoder])
    h_dec1 = activation(tf.matmul(z_cond, W_dec1) + b_dec1)

    W_dec2 = weight_variable([n_components_decoder, n_components_decoder])
    b_dec2 = bias_variable([n_components_decoder])
    h_dec2 = activation(tf.matmul(h_dec1, W_dec2) + b_dec2)

    W_dec3 = weight_variable([n_components_decoder, n_components_decoder])
    b_dec3 = bias_variable([n_components_decoder])
    h_dec3 = activation(tf.matmul(h_dec2, W_dec3) + b_dec3)

    W_mu_dec = weight_variable([n_components_decoder, n_features])
    b_mu_dec = bias_variable([n_features])
    y = tf.nn.sigmoid(tf.matmul(h_dec3, W_mu_dec) + b_mu_dec)

    # p(x|z)
    log_px_given_z = -tf.reduce_sum(
        x * tf.log(y + 1e-10) +
        (1 - x) * tf.log(1 - y + 1e-10), 1)

    # d_kl(q(z|x)||p(z))
    # Appendix B: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_div = -0.5 * tf.reduce_sum(
        1.0 + 2.0 * z_log_sigma - tf.square(z_mu) - tf.exp(2.0 * z_log_sigma),
        1)
    loss = tf.reduce_mean(log_px_given_z + kl_div)

    return {'cost': loss, 'x': x, 'z': z, 'y': y, 'cond':cond}


# %%
def test_mnist(result_dir='condVAE_Results'):
    """Summary

    Returns
    -------
    name : TYPE
        Description
    """
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
    ae = VAE()

    # %%
    learning_rate = 0.001
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
    test_xs, test_ys = mnist.test.next_batch(n_examples)
    xs, ys = mnist.test.images, mnist.test.labels
    fig_manifold, ax_manifold = plt.subplots(1, 1)
    fig_reconstruction, axs_reconstruction = plt.subplots(2, n_examples, figsize=(10, 2))
    fig_image_manifold, ax_image_manifold = plt.subplots(1, 1)
    for epoch_i in range(n_epochs):
        print('--- Epoch', epoch_i)
        train_cost = 0
        for batch_i in range(mnist.train.num_examples // batch_size):
            batch_xs, batch_cond = mnist.train.next_batch(batch_size)#get images and labels
            train_cost += sess.run([ae['cost'], optimizer],
                                   feed_dict={ae['x']: batch_xs,ae['cond']: batch_cond})[0]
    
        # %%
        # Plot example reconstructions from latent layer
        imgs = []
        dims_c = np.shape(batch_cond)
        rand_cond_ind = np.random.choice(dims_c[-1],1)#get one random cond for each plot
        rand_cond = np.zeros_like(batch_cond[0:1,:])
        rand_cond[0,rand_cond_ind] = 1. #assuming one-hot label representation
        for img_i in np.linspace(-3, 3, n_examples):
            for img_j in np.linspace(-3, 3, n_examples):
                z = np.array([[img_i, img_j]], dtype=np.float32)
                recon = sess.run(ae['y'], feed_dict={ae['z']: z,ae['cond']: rand_cond})
                imgs.append(np.reshape(recon, (1, 28, 28, 1)))
        imgs_cat = np.concatenate(imgs)
        ax_manifold.imshow(montage_batch(imgs_cat))
        fig_path = os.path.join(result_dir,'manifold_%08d.png' % t_i)
        fig_manifold.savefig(fig_path)

        # %%
        # Plot example reconstructions
        recon = sess.run(ae['y'], feed_dict={ae['x']: test_xs,ae['cond']: test_ys})
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
        zs = sess.run(ae['z'], feed_dict={ae['x']: xs, ae['cond']: ys})
        ax_image_manifold.clear()
        ax_image_manifold.scatter(zs[:, 0], zs[:, 1],
            c=np.argmax(ys, 1), alpha=0.2)
        ax_image_manifold.set_xlim([-6, 6])
        ax_image_manifold.set_ylim([-6, 6])
        ax_image_manifold.axis('off')
        fig_path = os.path.join(result_dir,'image_manifold_%08d.png' % t_i)
        fig_image_manifold.savefig(fig_path)

        t_i += 1


        print('Train cost:', train_cost /
              (mnist.train.num_examples // batch_size))

        valid_cost = 0
        for batch_i in range(mnist.validation.num_examples // batch_size):
            batch_xs, batch_cond = mnist.validation.next_batch(batch_size)
            valid_cost += sess.run([ae['cost']],
                                   feed_dict={ae['x']: batch_xs,ae['cond']: batch_cond})[0]
        print('Validation cost:', valid_cost /
              (mnist.validation.num_examples // batch_size))


if __name__ == '__main__':
    test_mnist()

