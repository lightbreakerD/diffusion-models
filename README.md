# Fun with Diffusion Models

We implement and deploy diffusion models for image generation. This is split into two
parts: sampling from a pretrained model, and training a diffusion model from scratch.

**Part A is in [`diffusion_sampling.ipynb`](diffusion_sampling.ipynb), Part B is in
[`diffusion_from_scratch.ipynb`](diffusion_from_scratch.ipynb).** There is a
`requirements.txt` for all the libraries we need. I have saved the results of the tasks
into the folders called `output` and `output2`.

## How does diffusion work?

If we have a clear image of something, we can progressively add noise to it every step.
Let's say we have a clean image at time step 0. We are going to add some noise at every
step. A diffusion model tries to reverse this process. It starts with a noisy image at
time step T and tries to remove the noise until it ends up with an image.

## Part A: The power of diffusion models

In part A we implement a diffusion sampling loop, play around with diffusion models and
use them for inpainting and creating optical illusions.

We are going to use the diffusion model DeepFloyd to generate our images using text input.
It is a two stage model by Stability AI. In the first stage it generates a 64x64 image.

I used different amounts of inference steps to generate the images. With only 2 inference
steps we do not get a good result, we end up with only noise. With 10 inference steps the
results are pretty good, we can see that the generated images are actually corresponding
to the prompts.

- **Forward process.** We take a clean image and add noise to it.
- **Classical denoising.** We can use the classical method of Gaussian blur filtering to
  remove the noise.
- **One step denoising.** The model we are using has a denoiser which has been trained on
  a very large image dataset.
- **Iterative denoising.** One step denoising performs not as good if we have more noise,
  which is why we want to denoise iteratively.
- **Classifier-free guidance, image-to-image translation, inpainting, visual anagrams and
  hybrid images.**

## Part B: Diffusion models from scratch

Training a single-step denoising UNet first, then a time-conditioned and class-conditioned
UNet with the DDPM forward and inverse process.

## Note

The notebook reads a Hugging Face token from the environment:

```bash
export HF_TOKEN=your_token_here
```
