#ifndef MNIST_LOAD
#define MNIST_LOAD

#define MAGIC_UNSIGNED_BYTE 0x08
#define MAGIC_IMAGES_DIM 3
#define MAGIC_LABELS_DIM 1
#define MNIST_LABEL_SIZE 10

#include <stdint.h>

#include "common.cuh"


/** @struct MNISTImage
 * Stores an MNIST-specific image.
 * 
 * @var MNISTImage::pixels
 *  The pixel values (within range [0, 255] inclusive) of the image stored in row-major format.
 * @var MNISTImage::height
 *  The height of the image.
 * @var MNISTImage::width
 *  The width of the image.
 */
typedef struct {
    uint8_t *pixels; 
    uint32_t height;
    uint32_t width;
} MNISTImage;


/** @struct MNISTDataset
 *  Stores an MNIST dataset that contains one or more images/labels.
 * 
 * @var MNISTDataset::images
 *  The one or more images stored in the dataset.
 * @var MNISTDataset::labels
 *  The labels corresponding to the images.
 * @var MNISTDataset::num_samples
 *  The number of samples.
 */
typedef struct {
    MNISTImage *images;
    uint8_t *labels;
    uint32_t num_samples;
} MNISTDataset;


/** @struct Image
 *  Stores a more generic image with float pixel values.
 * 
 * @var Image::pixels
 *  The pixel values of the image stored in row-major format.
 * @var Image::height
 *  The height of the image.
 * @var Image::width
 *  The width of the image.
 */
typedef struct {
    float *pixels;
    uint32_t height;
    uint32_t width;
} Image;


/** @struct ImageDataset
 *  Stores a generic image dataset that contains one or more images/labels.
 * 
 * @var ImageDataset::images
 *  The one or more images stored in the dataset.
 * @var ImageDataset::labels
 *  The labels corresponding to the images.
 * @var ImageDataset::num_samples
 *  The number of samples.
 * @var ImageDataset::view_indices
 *  The indices of the object's view; it allows shuffling the indices of the samples
 *  without having to modify the data directly.
 */
typedef struct {
    Image *images;
    uint8_t *labels;
    uint32_t num_samples;
    uint32_t *view_indices;
} ImageDataset;


/**
 * Deallocates MNIST images.
 * 
 * @param images The corresponding MNIST images to be deallocated.
 * @param count The number of MNIST images to be allocated.
 */
void free_MNIST_images(MNISTImage *images, uint32_t count);


/**
 * Deallocates an MNIST dataset.
 * 
 * @param dataset The corresponding MNIST dataset to be deallocated.
 */
void free_MNIST_dataset(MNISTDataset *dataset);


/**
 * Deallocates images.
 * 
 * @param images The corresponding images to be deallocated.
 * @param count The number of images to be allocated.
 */
void free_images(Image *images, uint32_t count);


/**
 * Deallocates an image dataset.
 * 
 * @param dataset The corresponding dataset to be deallocated.
 */
void free_dataset(ImageDataset *dataset);


/**
 * Loads an MNIST dataset given image and label files.
 * 
 * @param images_file_path The path to image idx file.
 * @param labels_file_path The path to label idx file.
 * @return The loaded MNIST dataset.
 */
MNISTDataset *load_mnist_dataset(const char *images_file_path, const char *labels_file_path);


/**
 * Shuffles the indices of the samples within the dataset by shuffling only the `view_indices`.
 * 
 * @param dataset The image dataset input.
 * @param seed The randomization seed.
 */
void shuffle_indices(ImageDataset *dataset, uint8_t seed);


/**
 * Splits the dataset given range.
 * 
 * @param dataset The image dataset input.
 * @param begin_index The starting index.
 * @param end_index The end index (not inclusive).
 * @param release Whether to release the pointer of the inputted dataset to the images; this is used when
 *  we want to release the dataset early after splitting is completed.
 * @return The splitted/cropped dataset.
 */
ImageDataset *split_dataset(ImageDataset *dataset, uint32_t begin_index, uint32_t end_index, bool release_images);


/**
 * Prepares a batch by storing the images and labels in float and uint8_t arrays, respectively.
 * 
 * @param X The array to store the images' pixel values in row-major format.
 * @param y The array to store one-hot encodings of the labels corresponding to the images.
 * @param dataset The dataset the batch is prepared from.
 * @param start_index The starting index of the samples in the batch.
 * @param num_samples_in_batch The number of samples to be stored in the batch.
 */
void prepare_batch(
    float X[], uint8_t y[],
    ImageDataset *dataset,
    uint32_t start_index, uint32_t num_samples_in_batch
);

#endif