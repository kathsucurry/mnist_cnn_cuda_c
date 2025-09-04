#ifndef PREPROCESSING
#define PREPROCESSING

#include <stdint.h>
#include "data_loader.cuh"


/**
 * Generates a copy of (ImageDataset *) out of (MNISTDataset *) to be used for the model training. 
 * 
 * @param dataset The MNIST-specific dataset to be converted into ImageDataset.
 * @return The generated (ImageDataset *).
 */
ImageDataset *generate_image_dataset(MNISTDataset *dataset);


/**
 * Normalizes the pixel values of the images in (ImageDataset *). Divides the pixel values by 255.0.
 * 
 * @param dataset The (ImageDataset *) input.
 */
void normalize_pixels(ImageDataset *dataset);


/**
 * Adds a number of zero padding ot the images in (ImageDataset *).
 * 
 * @param dataset The (ImageDataset *) input.
 * @param num_padding The number of zero padding to be added to each side of each image.
 */
void add_padding(ImageDataset *dataset, uint8_t num_padding);


#endif