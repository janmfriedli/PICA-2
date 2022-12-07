# PICA-2
Abstract AI Art via Computational Creativity using Deep Convolutional Generative Adverserial Networks (DCGANs)

## What are DCGANs?

In short, DCGANs are sequential models, that make use of both a Generator (that generates a new image) and a Discriminator (that classifies to which degree the generated picture belongs to a class).

For a detailed example of a DCGAN, go to https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/dcgan.ipynb

## How do we make use of DCGANs to create AI art?

For our model, we make use of the same setup given in the page linked above. We make use of the Quick, Draw! dataset (https://quickdraw.withgoogle.com/data). The idea is as follows:

1. Train different DCGANs, one for each category. This results in multiple models, each of which can create new drawings within a category, looking similar to the ones in the dataset.

2. Depending on the user input, generate one image per input category using the generator of the trained model belonging to the respective category.

3. The next step, is to use the discriminator of both models to classify the degree of 'fakeness'. The generated picture of the first image, let's say 'apple', will be our starting point. We will classify the fakeness of the generated apple two times, using once the discriminator of the apple model (likely to be real, discriminator loss value close to 0), and once using the discriminator of the model of the second input, let's say house (likely to be very fake, discriminator loss value far below zero).

4. We want to generate abstract art based on two input categories. Hence, we do not want any of the discriminators to classify the image as 'real'. So, because it looks to much like an apple, we add a fraction ('strength' parameter) of the generated house on top of the apple. After this is done, we repeat step 3.

5. Steps 3. and 4. wille be repeated until a condition is met. This condition is basically that the discriminator loss value of both the apple and house model has to be below a certain point. Because we start with an apple and add houses on top, it will still have some characteristics visible from both classes when it reaches this amount of abstractness.

6. Add a color mapping on top, and a signature on the image.

The code for performing these steps can be found in the fast.py file.
