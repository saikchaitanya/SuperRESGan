from keras.layers import Input, Dense, DepthwiseConv2D
from keras.layers import BatchNormalization, Activation, Add
from keras.layers import LeakyReLU
from keras.layers import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.models import Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
from dataloader import DataLoader
from argparse import ArgumentParser
import numpy as np
import os

import keras.backend as K

parser = ArgumentParser()
parser.add_argument('--image_dir', type=str, help='Path to high resolution image directory')
parser.add_argument('--batch_size', type=int, help='Batch size for training')
parser.add_argument('--epochs', type=int, help='Batch size for training')


class SRGAN(object):
    def __init__(self, data_path):
        # Input shape
        self.channels = 3
        self.lr_height = 128  # Low resolution height
        self.lr_width = 128  # Low resolution width
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        self.hr_height = self.lr_height * 4  # High resolution height
        self.hr_width = self.lr_width * 4  # High resolution width
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        # Number of residual blocks in the generator
        self.n_residual_blocks = 14

        optimizer = Adam(1e-4)

        # We use a pre-trained VGG19 model to extract image features from the high resolution
        # and the generated high resolution images and minimize the mse between them
        self.vgg = self.build_vgg()
        self.vgg.trainable = False
        self.vgg.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])

        # Configure data loader
        self.dataset_name = data_path
        self.data_loader = DataLoader(image_dir=self.dataset_name,
                                      img_res=(self.hr_height, self.hr_width))

        # Calculate output shape of D (PatchGAN)
        patch = int(self.hr_height / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 8
        self.df = 64

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='mse',
                               optimizer=optimizer)

        # High res. and low res. images
        img_hr = Input(shape=self.hr_shape)
        img_lr = Input(shape=self.lr_shape)

        # Generate high res. version from low res.
        fake_hr = self.generator(img_lr)

        # Extract image features of the generated img
        fake_features = self.vgg(fake_hr)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # Discriminator determines validity of generated high res. images
        validity = self.discriminator(fake_hr)

        self.combined = Model([img_lr, img_hr], [validity, fake_features])
        self.combined.compile(loss=['binary_crossentropy', 'mse'],
                              loss_weights=[1e-3, 1],
                              optimizer=optimizer)

    def build_vgg(self):
        """
        Builds a pre-trained VGG19 model that outputs image features extracted at the
        third block of the model
        """
        # Input image to extract features from
        img = Input(shape=self.hr_shape)

        # Get the vgg network. Extract features from last conv layer
        vgg = VGG19(weights="imagenet", include_top=False)
        vgg.outputs = [vgg.layers[20].output]

        # Create model and compile
        model = Model(inputs=img, outputs=vgg(img))
        model.trainable = False
        return model

    def build_generator(self):

        def _make_divisible(v, divisor, min_value=None):
            if min_value is None:
                min_value = divisor
            new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
            # Make sure that round down does not go down by more than 10%.
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v

        def residual_block(inputs, filters, block_id, expansion=6, stride=1, alpha=1.0):
            channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

            in_channels = K.int_shape(inputs)[channel_axis]
            pointwise_conv_filters = int(filters * alpha)
            pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
            x = inputs
            prefix = 'block_{}_'.format(block_id)

            if block_id:
                # Expand
                x = Conv2D(expansion * in_channels,
                           kernel_size=1,
                           padding='same',
                           use_bias=False,
                           activation=None,
                           name=prefix + 'expand')(x)
                x = BatchNormalization(axis=channel_axis,
                                       epsilon=1e-3,
                                       momentum=0.999,
                                       name=prefix + 'expand_BN')(x)
                x = Activation('relu', name=prefix + 'expand_relu')(x)
            else:
                prefix = 'expanded_conv_'

            # Depthwise
            x = DepthwiseConv2D(kernel_size=3,
                                strides=stride,
                                activation=None,
                                use_bias=False,
                                padding='same' if stride == 1 else 'valid',
                                name=prefix + 'depthwise')(x)
            x = BatchNormalization(axis=channel_axis,
                                   epsilon=1e-3,
                                   momentum=0.999,
                                   name=prefix + 'depthwise_BN')(x)

            x = Activation('relu', name=prefix + 'depthwise_relu')(x)

            # Project
            x = Conv2D(pointwise_filters,
                       kernel_size=1,
                       padding='same',
                       use_bias=False,
                       activation=None,
                       name=prefix + 'project')(x)
            x = BatchNormalization(axis=channel_axis,
                                   epsilon=1e-3,
                                   momentum=0.999,
                                   name=prefix + 'project_BN')(x)

            if in_channels == pointwise_filters and stride == 1:
                return Add(name=prefix + 'add')([inputs, x])
            return x

        def deconv2d(layer_input):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(self.gf, kernel_size=3, strides=1, padding='same')(u)
            u = Activation('relu')(u)
            return u

        # Low resolution image input
        img_lr = Input(shape=self.lr_shape)

        # Pre-residual block
        c1 = Conv2D(self.gf, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = Activation('relu')(c1)

        # Propogate through residual blocks
        r = residual_block(c1, self.gf, 0)
        for idx in range(1, self.n_residual_blocks):
            r = residual_block(r, self.gf, idx)

        # Post-residual block
        c2 = Conv2D(self.gf, kernel_size=3, strides=1, padding='same')(r)
        c2 = BatchNormalization(momentum=0.8)(c2)
        c2 = Add()([c2, c1])

        # Upsampling
        u1 = deconv2d(c2)
        u2 = deconv2d(u1)

        # Generate high resolution output
        gen_hr = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)

        return Model(img_lr, gen_hr)

    def build_discriminator(self):

        def d_block(layer_input, filters, strides=1, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Input img
        d0 = Input(shape=self.hr_shape)

        d1 = d_block(d0, self.df, bn=False)
        d2 = d_block(d1, self.df, strides=2)
        d3 = d_block(d2, self.df * 2)
        d4 = d_block(d3, self.df * 2, strides=2)
        d5 = d_block(d4, self.df * 4)
        d6 = d_block(d5, self.df * 4, strides=2)
        d7 = d_block(d6, self.df * 8)
        d8 = d_block(d7, self.df * 8, strides=2)

        d9 = Dense(self.df * 16)(d8)
        d10 = LeakyReLU(alpha=0.2)(d9)
        validity = Dense(1, activation='sigmoid')(d10)

        return Model(d0, validity)

    def train(self, epochs, batch_size=1, sample_interval=200):

        # Pre-train the generator so that it can fool the discriminator at the start.
        for i in range(400):
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)
            self.generator.train_on_batch(imgs_lr, imgs_hr)

        # GAN training
        for epoch in range(epochs):

            # ----------------------
            #  Train Discriminator
            # ----------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            # From low res. image generate high res. version
            fake_hr = self.generator.predict(imgs_lr)

            valid = np.ones((batch_size,) + self.disc_patch) - np.random.random_sample(
                (batch_size,) + self.disc_patch) * 0.2
            fake = np.zeros((batch_size,) + self.disc_patch) * np.random.random_sample(
                (batch_size,) + self.disc_patch) * 0.2

            # Train the discriminators (original images = real / generated = Fake)
            d_loss_real = self.discriminator.train_on_batch(imgs_hr, valid)
            d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ------------------
            #  Train Generator
            # ------------------

            # Sample images and their conditioning counterparts
            imgs_hr, imgs_lr = self.data_loader.load_data(batch_size)

            # The generators want the discriminators to label the generated images as real
            valid = np.ones((batch_size,) + self.disc_patch) - np.random.random_sample(
                (batch_size,) + self.disc_patch) * 0.2

            # Extract ground truth image features using pre-trained VGG19 model
            image_features = self.vgg.predict(imgs_hr)

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_lr, imgs_hr], [valid, image_features])

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print("%d time: %s" % (epoch, elapsed_time))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                print("g_loss: {}, d_loss: {}".format(g_loss, d_loss))
                self.sample_images(epoch)
                self.generator.save('gen.h5')

    def sample_images(self, epoch):
        os.makedirs('images', exist_ok=True)
        r, c = 2, 2

        imgs_hr, imgs_lr = self.data_loader.load_data(batch_size=2, is_testing=True)
        fake_hr = self.generator.predict(imgs_lr)

        # Rescale images 0 - 1
        imgs_lr = 0.5 * imgs_lr + 0.5
        fake_hr = 0.5 * fake_hr + 0.5
        imgs_hr = 0.5 * imgs_hr + 0.5

        # Save generated images and the high resolution originals
        titles = ['Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for row in range(r):
            for col, image in enumerate([fake_hr, imgs_hr]):
                axs[row, col].imshow(image[row])
                axs[row, col].set_title(titles[col])
                axs[row, col].axis('off')
            cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()

        # Save low resolution images for comparison
        for i in range(r):
            fig = plt.figure()
            plt.imshow(imgs_lr[i])
            fig.savefig('images/%d_lowres%d.png' % (epoch, i))
            plt.close()


if __name__ == '__main__':
    args = parser.parse_args()
    gan = SRGAN(args.image_dir)
    gan.train(epochs=args.epochs, batch_size=args.batch_size, sample_interval=200)
