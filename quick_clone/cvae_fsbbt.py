# https://github.com/nphdang/FS-BBT/blob/main/cifar10/cvae.py

from typing import List

import tensorflow as tf
keras = tf.keras

if __name__ == '__main__':
    import os, sys
    repo_path = os.path.abspath(os.path.join(__file__, '../..'))
    assert os.path.basename(repo_path) == 'kd_tf', "Wrong parent folder. Please change to 'kd_tf'"
    sys.path.append(repo_path)

    from models.GANs.CGAN import CGAN
    from models.GANs.utils import MakeConditionalSyntheticGIFCallback, MakeInterpolateSyntheticGIFCallback
    from dataloader import dataloader
else:
    from ..models.GANs.CGAN import CGAN
    from ..models.GANs.utils import MakeConditionalSyntheticGIFCallback, MakeInterpolateSyntheticGIFCallback
    from ..dataloader import dataloader

# train CVAE
# Encoder: q(z|x)

def define_conv_discriminator(image_dim:List[int]=[28, 28, 1],
                         base_dim:List[int]=[7, 7, 256],
                         num_classes:int=10,
                         **kwargs):
    # Parse architecture from input dimension
    dim_ratio = [image_dim[axis]/base_dim[axis] for axis in range(len(image_dim)-1)]
    for axis in range(len(dim_ratio)):
        num_conv = tf.math.log(dim_ratio[axis])/tf.math.log(2.)
        assert num_conv == int(num_conv), f'`base_dim` {base_dim[axis]} is not applicable with `image_dim` {image_dim[axis]} at axis {axis}.'
        assert dim_ratio[axis] == dim_ratio[0], f'Ratio of `image_dim` and `base_dim` {image_dim[axis]}/{base_dim[axis]} at axis {axis} is mismatched with {image_dim[0]}/{base_dim[0]} at axis 0.'
    num_conv = int(num_conv)
    
    image_input = keras.layers.Input(shape=image_dim)

    label_input = keras.layers.Input(shape=[num_classes])
    label_branch = keras.layers.Dense(units=tf.reduce_prod(image_dim[0:-1]))(label_input)
    label_branch = keras.layers.Reshape(target_shape=[*image_dim[0:-1], 1])(label_branch)

    main_branch = keras.layers.Concatenate()([image_input, label_branch])
    for i in range(num_conv):
        filters = base_dim[-1] // 2**(num_conv-1-i)
        main_branch = keras.layers.Conv2D(filters=filters, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu')(main_branch)
    main_branch = keras.layers.Flatten()(main_branch)
    outputs = keras.layers.Dense(units=1, activation='sigmoid')(main_branch)

    disc = keras.Model(
        inputs=[image_input, label_input],
        outputs=outputs,
        **kwargs
    )
    disc.build(input_shape=[[None, *image_dim], [None, num_classes]])
    return disc

def define_conv_generator(latent_dim:int=128,
                          image_dim:List[int]=[28, 28, 1],
                          base_dim:List[int]=[7, 7, 256],
                          num_classes:int=10,
                          **kwargs):
    # Parse architecture from input dimension
    dim_ratio = [image_dim[axis]/base_dim[axis] for axis in range(len(image_dim)-1)]
    for axis in range(len(dim_ratio)):
        num_conv = tf.math.log(dim_ratio[axis])/tf.math.log(2.)
        assert num_conv == int(num_conv), f'`base_dim` {base_dim[axis]} is not applicable with `image_dim` {image_dim[axis]} at axis {axis}.'
        assert dim_ratio[axis] == dim_ratio[0], f'Ratio of `image_dim` and `base_dim` {image_dim[axis]}/{base_dim[axis]} at axis {axis} is mismatched with {image_dim[0]}/{base_dim[0]} at axis 0.'
    num_conv = int(num_conv)

    latent_input = keras.layers.Input(shape=[latent_dim])
    label_input = keras.layers.Input(shape=[num_classes])
    main_branch = keras.layers.Concatenate()([latent_input, label_input])

    main_branch = keras.layers.Dense(units=16, activation='relu')(main_branch)
    main_branch = keras.layers.Dense(units=32, activation='relu')(main_branch)
    main_branch = keras.layers.Dense(units=tf.reduce_prod(base_dim))(main_branch)
    main_branch = keras.layers.Reshape(target_shape=base_dim)(main_branch)
    for i in range(num_conv):
        filters = base_dim[-1] // 2**(i+1)
        if i < num_conv - 1:
            main_branch = keras.layers.Conv2DTranspose(filters=filters, kernel_size=(4, 4), strides=(2, 2), padding='same', activation='relu')(main_branch)
        elif i == num_conv - 1:
            outputs = keras.layers.Conv2DTranspose(filters=image_dim[-1], kernel_size=(4, 4), strides=(2, 2), padding='same', activation='tanh')(main_branch)

    gen = keras.Model(
        inputs=[latent_input, label_input],
        outputs=outputs,
        **kwargs
    )
    gen.build(input_shape=[[None, latent_dim], [None, num_classes]])
    return gen

    # x_encoded = Input(shape=(latent + n_class,))
    # h_p = Dense(16, activation='relu')(x_encoded)
    # h_p = Dense(32, activation='relu')(h_p)
    # h = Dense(q_shape[1] * q_shape[2] * q_shape[3])(h_p)
    # p = Reshape(target_shape=(q_shape[1], q_shape[2], q_shape[3]))(h)
    # p = Conv2DTranspose(filters=512, kernel_size=4, strides=2, padding='same', activation='relu')(p)
    # p = Conv2DTranspose(filters=256, kernel_size=4, strides=2, padding='same', activation='relu')(p)
    # p = Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same', activation='relu')(p)
    # p = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same', activation='relu')(p)
    # p = Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='same', activation='relu')(p)
    # p = Conv2DTranspose(filters=3, kernel_size=4, strides=2, padding='same', activation='relu')(p)
    # flat = Flatten()(p)
    # x_decoded = Dense(n_feature, activation='sigmoid')(flat)
    # decoder = Model(x_encoded, x_decoded, name="decoder")

def experiment_mnist(latent_dim:int=16):
    image_dim = [28, 28, 1]
    num_classes = 10

    latent_input = keras.layers.Input(shape=[latent_dim])
    label_input = keras.layers.Input(shape=[num_classes])
    main_branch = keras.layers.Concatenate()([latent_input, label_input])
    main_branch = keras.layers.Dense(64, activation='relu')(main_branch)
    main_branch = keras.layers.Dense(128, activation='relu')(main_branch)
    main_branch = keras.layers.Dense(256, activation='relu')(main_branch)
    main_branch = keras.layers.Dense(units=tf.reduce_prod(image_dim), activation='sigmoid')(main_branch)
    outputs = keras.layers.Reshape(target_shape=image_dim)(main_branch)

    cgen = keras.Model(inputs=[latent_input, label_input], outputs=outputs, name='DenseCGen')
    cgen.build(input_shape=[[None, latent_dim], [None, num_classes]])


    image_input = keras.layers.Input(shape=image_dim)
    label_input = keras.layers.Input(shape=[num_classes])
    image_branch = keras.layers.Flatten()(image_input)
    main_branch = keras.layers.Concatenate()([image_branch, label_input])
    main_branch = keras.layers.Dense(256, activation='relu')(main_branch)
    main_branch = keras.layers.Dense(128, activation='relu')(main_branch)
    main_branch = keras.layers.Dense(64, activation='relu')(main_branch)
    outputs = keras.layers.Dense(1, activation='sigmoid')(main_branch)

    cdisc = keras.Model(inputs=[image_input, label_input], outputs=outputs, name='DenseCDisc')
    cdisc.build(input_shape=[[None, *image_dim], [None, num_classes]])

    cgan = CGAN(
        generator=cgen,
        discriminator=cdisc,
        latent_dim=latent_dim,
        image_dim=image_dim,
        embed_dim=-1,
        num_classes=num_classes,
        onehot_input=True
    )
    cgan.build()
    cgan.summary(with_graph=True, expand_nested=True, line_length=120)
    cgan.compile()

    ds, info = dataloader(
        dataset='mnist',
        batch_size_train=64,
        batch_size_test=1000,
        drop_remainder=True,
        onehot_label=True,
        with_info=True)
    class_names = info.features['label'].names


    csv_logger = keras.callbacks.CSVLogger(
        f'./logs/{cgan.name}_{cgan.generator.name}_{cgan.discriminator.name}.csv',
        append=True)
    
    gif_maker = MakeConditionalSyntheticGIFCallback(
        filename=f'./logs/{cgan.name}_{cgan.generator.name}_{cgan.discriminator.name}.gif', 
        postprocess_fn=lambda x:(x+1)/2,
        class_names=class_names
    )
    slerper = MakeInterpolateSyntheticGIFCallback(
        filename=f'./logs/{cgan.name}_{cgan.generator.name}_{cgan.discriminator.name}_itpl_slerp.gif',
        itpl_method='slerp',
        postprocess_fn=lambda x:(x+1)/2,
        class_names=class_names
    )
    cgan.fit(
        x=ds['train'],
        epochs=50,
        callbacks=[csv_logger, gif_maker, slerper],
        validation_data=ds['test'],
    )

if __name__ == '__main__':
    # cdisc = define_conv_discriminator(
    #     image_dim=[28, 28, 1],
    #     base_dim=[7, 7, 256],
    #     num_classes=10,
    #     name='cDisc'
    # )
    # cdisc.summary(expand_nested=True, line_length=120)

    # cgen = define_conv_generator(
    #     latent_dim=128,
    #     image_dim=[28, 28, 1],
    #     base_dim=[7, 7, 256],
    #     num_classes=10,
    #     name='cGen'
    # )
    # cgen.summary(expand_nested=True, line_length=120)

    experiment_mnist(latent_dim=2)
