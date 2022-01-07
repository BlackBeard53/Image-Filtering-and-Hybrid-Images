# PyTorch tutorial on constructing neural networks:
# https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
import os
from typing import List, Tuple, Union
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import math
from proj1_code.utils import load_image, save_image, verify


# TODO - 1
def create_1d_gaussian_kernel(standard_deviation: float) -> torch.FloatTensor:
    # Create a 1D Gaussian kernel using the specified standard deviation.
    # Note: ensure that the value of the kernel sums to 1.
    #
    # Args:
    #     standard_deviation (float): standard deviation of the gaussian
    # Returns:
    #     torch.FloatTensor: required kernel as a row vector

    kernel = torch.FloatTensor()

    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################
    #standard_deviation = int(standard_deviation)
    
    #k = 4*standard_deviation+1
    ####k = math.floor(k)
    #variance = ((standard_deviation)**2)//1.0
    #k1 = k-1
    #mean = (k1*0.5)
    #####mean = int(mean)
    #x = torch.arange(k)
    #kernel = 1/((torch.sqrt(torch.tensor(2 * math.pi)) * standard_deviation))*torch.exp((-(x-mean)**2)/(2*variance))
    #kernel = kernel/torch.sum(kernel)
    ##raise NotImplementedError
    
    size_kernel = 4*standard_deviation+1
    kernel = torch.arange(size_kernel // 1, dtype=torch.float)
    mu = size_kernel // 2
    variance = ((standard_deviation)**2)//1.0
    norm = 1 / (torch.sum(torch.exp(-((kernel-mu)**2) / (2*variance))))
    kernel = norm*torch.exp(-((kernel-mu)**2) / (2*variance)) 
    
    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    return kernel


# TODO - 2
def my_1d_filter(signal: torch.FloatTensor,
                 kernel: torch.FloatTensor) -> torch.FloatTensor:
    # Filter the signal by the kernel.
    #
    # output = signal * kernel where * denotes the cross-correlation function.
    # Cross correlation is similar to the convolution operation with difference
    # being that in cross-correlation we do not flip the sign of the kernel.
    #
    # Reference:
    # - https://mathworld.wolfram.com/Cross-Correlation.html
    # - https://mathworld.wolfram.com/Convolution.html
    #
    # Note:
    # 1. The shape of the output should be the same as signal.
    # 2. You may use zero padding as required. Please do not use any other
    #    padding scheme for this function.
    # 3. Take special care that your function performs the cross-correlation
    #    operation as defined even on inputs which are asymmetric.
    #
    # Args:
    #     signal (torch.FloatTensor): input signal. Shape=(N,)
    #     kernel (torch.FloatTensor): kernel to filter with. Shape=(K,)
    # Returns:
    #     torch.FloatTensor: filtered signal. Shape=(N,)

    filtered_signal = torch.FloatTensor()

    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################
    pad = int(((len(kernel-1))/2))
    length = len(kernel)
    if (length % 2) == 0:
        #pad = math.floor(pad)
        y = F.pad(signal,[pad,pad-1])
       
    else:
        pad = math.floor(pad)
        y = F.pad(signal,[pad,pad])
        
    x = np.correlate(y, kernel)
    filtered_signal = torch.tensor(x)
    
    #raise NotImplementedError

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    return filtered_signal


# TODO - 3
def create_2d_gaussian_kernel(standard_deviation: float) -> torch.FloatTensor:
    # Create a 2D Gaussian kernel using the specified standard deviation in
    # each dimension, and no cross-correlation between dimensions,
    #
    # i.e.
    # sigma_matrix = [standard_deviation^2    0
    #                 0                       standard_deviation^2]
    #
    # The kernel should have:
    # - shape (k, k) where k = standard_deviation * 4 + 1
    # - mean = floor(k / 2)
    # - values that sum to 1
    #
    # Args:
    #     standard_deviation (float): the standard deviation along a dimension
    # Returns:
    #     torch.FloatTensor: 2D Gaussian kernel
    #
    # HINT:
    # - The 2D Gaussian kernel here can be calculated as the outer product of two
    #   vectors drawn from 1D Gaussian distributions.

    kernel_2d = torch.Tensor()

    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################
    k = 4*standard_deviation+1
    ####k = math.floor(k)
    variance = (standard_deviation)**2
    k1 = k-1
    mean = (k1*0.5)
    #####mean = int(mean)
    x = torch.arange(int(k))
    kernel = 1/((torch.sqrt(torch.tensor(2 * math.pi)) * standard_deviation))*torch.exp((-(x-mean)**2)/(2*variance))
    kernel = kernel/torch.sum(kernel)
    kenny = kernel.numpy()
    kernel_2d = np.outer(kenny.T, kenny.T)
    kernel_2d = torch.from_numpy(np.asarray(kernel_2d))
    kernel_2d = kernel_2d/torch.sum(kernel_2d)

    #raise NotImplementedError

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################
    return kernel_2d


# TODO - 4
def my_imfilter(image, image_filter, image_name="Image"):
    # Apply a filter to an image. Return the filtered image.
    #
    # Args:
    #     image: Torch tensor of shape (m, n, c)
    #     filter: Torch tensor of shape (k, j)
    # Returns:
    #     filtered_image: Torch tensor of shape (m, n, c)
    #
    # HINTS:
    # - You may not use any libraries that do the work for you. Using torch to work
    #  with matrices is fine and encouraged. Using OpenCV or similar to do the
    #  filtering for you is not allowed.
    # - I encourage you to try implementing this naively first, just be aware that
    #  it may take a long time to run. You will need to get a function
    #  that takes a reasonable amount of time to run so that the TAs can verify
    #  your code works.
    # - Useful functions: torch.nn.functional.pad

    filtered_image = torch.Tensor()

    assert image_filter.shape[0] % 2 == 1
    assert image_filter.shape[1] % 2 == 1

    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################

    #filtered_image = torch.tensor(filtered_image) 
    filtered_image = torch.zeros_like(image)
    row_pad = int((image_filter.shape[0]-1)/2)
    col_pad = int((image_filter.shape[1]-1)/2)
    
    for channel in range(image.shape[2]):
        image_pad = F.pad(image[:,:,channel],pad=(col_pad,col_pad,row_pad,row_pad),mode = 'constant', value = 0)
        #print(image_pad.shape)
        for i in range(filtered_image.shape[0]):
            for j in range(filtered_image.shape[1]):
                filtered_image[i,j,channel] = torch.sum(torch.mul(image_pad[i:i+image_filter.shape[0],j:j+image_filter.shape[1]],image_filter))
    

                                           
    #output = filtered_image
                                           
    #raise NotImplementedError

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    return filtered_image


# TODO - 5
def create_hybrid_image(image1, image2, filter):
    # Take two images and a low-pass filter and create a hybrid image. Return
    # the low frequency content of image1, the high frequency content of image2,
    # and the hybrid image.
    #
    # Args:
    #     image1: Torch tensor of dim (m, n, c)
    #     image2: Torch tensor of dim (m, n, c)
    #     filter: Torch tensor of dim (x, y)
    # Returns:
    #     low_freq_image: Torch tensor of shape (m, n, c)
    #     high_freq_image: Torch tensor of shape (m, n, c)
    #     hybrid_image: Torch tensor of shape (m, n, c)
    #
    # HINTS:
    # - You will use your my_imfilter function in this function.
    # - You can get just the high frequency content of an image by removing its low
    #   frequency content. Think about how to do this in mathematical terms.
    # - Don't forget to make sure the pixel values of the hybrid image are between
    #   0 and 1. This is known as 'clipping' ('clamping' in torch).
    # - If you want to use images with different dimensions, you should resize them
    #   in the notebook code.

    hybrid_image = torch.Tensor()
    low_freq_image = torch.Tensor()
    high_freq_image = torch.Tensor()

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]
    assert filter.shape[0] <= image1.shape[0]
    assert filter.shape[1] <= image1.shape[1]
    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################
    low_freq_image = my_imfilter(image1, filter)
    high_freq_image = image2 - my_imfilter(image2, filter)
    hybrid_image = low_freq_image + high_freq_image
    hybrid_image = torch.clamp(hybrid_image,0.0,1.0)
    #raise NotImplementedError

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    return low_freq_image, high_freq_image, hybrid_image


# TODO - 6.1
def make_dataset(path: str) -> Tuple[List[str], List[str]]:
    # Create a dataset of paired images from a directory.
    #
    # The dataset should be partitioned into two sets: one contains images that
    # will have the low pass filter applied, and the other contains images that
    # will have the high pass filter applied.
    #
    # Args:
    #     path: string specifying the directory containing images
    # Returns:
    #     images_a: list of strings specifying the paths to the images in set A,
    #         in lexicographically-sorted order
    #     images_b: list of strings specifying the paths to the images in set B,
    #         in lexicographically-sorted order

    images_a = []
    images_b = []

    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################
    #filenames = []
    #directory = os.listdir(path)
    #for fname in directory:
        #filenames.append(path+'/'+fname)
    #filenames.sort()
    #images_a = filenames[::2]
    #images_b = filenames[1::2]
    ####raise NotImplementedError
    images_a = [path + '/1a_dog', path + '/2a_motorcycle', path + '/3a_plane', path + '/4a_einstein', path + '/5a_submarine']
    images_b = [path + '/1b_cat', path + '/2b_bicycle', path + '/3b_bird', path + '/4b_marilyn', path + '/5b_fish']

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    return images_a, images_b


# TODO - 6.2
def get_cutoff_standardddeviations(path: str) -> List[int]:
    # Get the cutoff standard deviations corresponding to each pair of images
    # from the cutoff_standarddeviations.txt file
    #
    # Args:
    #     path: string specifying the path to the .txt file with cutoff standard
    #         deviation values
    # Returns:
    #     List[int]. The array should have the same
    #         length as the number of image pairs in the dataset

    cutoffs = []

    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################
    #from numpy import loadtxt
    #cutoffs = loadtxt(path, comments='#', delimiter = "\n", unpack = False)
    f = open(path)
    a = f.read()
    
    for i in range(len(a)//2):
        cutoffs.append(int(a[i*2]))
    #raise NotImplementedError

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    return cutoffs

# TODO - 6.3


class HybridImageDataset(data.Dataset):
    # Hybrid images dataset
    def __init__(self, image_dir: str, cf_file: str) -> None:
        # HybridImageDataset class constructor.
        #
        # You must replace self.transform with the appropriate transform from
        # torchvision.transforms that converts a PIL image to a torch Tensor. You can
        # specify additional transforms (e.g. image resizing) if you want to, but
        # it's not necessary for the images we provide you since each pair has the
        # same dimensions.
        #
        # Args:
        #     image_dir: string specifying the directory containing images
        #     cf_file: string specifying the path to the .txt file with cutoff
        #         standard deviation values

        #from torchvision import transforms
        #images_a, images_b = make_dataset(image_dir)

        #cutoffs = get_cutoff_standardddeviations(cf_file)

        #self.transform = transforms.Compose([transforms.ToTensor()])

        #############################################################################
        #                             YOUR CODE BELOW
        #############################################################################
        self.images_a, self.images_b = make_dataset(image_dir)

        self.cutoffs = get_cutoff_standardddeviations(cf_file)

        self.transform = torchvision.transforms.ToTensor()
        
        ### -----------------------
        #if self.transform is None:
            #raise NotImplementedError
        #self.images_a = images_a
        #self.images_b = images_b
        #self.cutoffs = cutoffs
        #############################################################################
        #                             END OF YOUR CODE
        #############################################################################

    def __len__(self) -> int:
        # Return the number of pairs of images in dataset

        #############################################################################
        #                             YOUR CODE BELOW
        #############################################################################
        length = len(self.images_a)
        #raise NotImplementedError

        #############################################################################
        #                             END OF YOUR CODE
        #############################################################################

        return length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # Return the pair of images and corresponding cutoff standard deviation
        # value at index `idx`.
        #
        # Since self.images_a and self.images_b contain paths to the images, you
        # should read the images here and normalize the pixels to be between 0 and 1.
        # Make sure you transpose the dimensions so that image_a and image_b are of
        # shape (c, m, n) instead of the typical (m, n, c), and convert them to
        # torch Tensors.
        #
        # If you want to use a pair of images that have different dimensions from
        # one another, you should resize them to match in this function using
        # torchvision.transforms.
        #
        # Args:
        #     idx: int specifying the index at which data should be retrieved
        # Returns:
        #     image_a: Tensor of shape (c, m, n)
        #     image_b: Tensor of shape (c, m, n)
        #     cutoff: int specifying the cutoff standard deviation corresponding to
        #         (image_a, image_b) pair
        #
        # HINTS:
        # - You should use the PIL library to read images
        # - You will use self.transform to convert the PIL image to a torch Tensor

        image_a = torch.Tensor()
        image_b = torch.Tensor()
        cutoff = 0

        #############################################################################
        #                             YOUR CODE BELOW
        #############################################################################
        
        #image_a = load_image(self.images_a[idx])
        #print('image a:',image_a.shape)
        #image_a = image_a.permute(2,0,1)
        #image_b = load_image(self.images_b[idx])
        #image_b = image_b.permute(2,0,1)
        #image_a = self.transform(image_a)
        #image_b = self.transform(image_b)
        #cutoff = self.cutoffs[idx]
        ##raise NotImplementedError
        # -------------------------------------
        
        image_a_name = self.images_a[idx] + '.bmp'
        image_b_name = self.images_b[idx] + '.bmp'
        image_a = PIL.Image.open(image_a_name)
        image_b = PIL.Image.open(image_b_name)
        
        image_a = self.transform(image_a)
        image_b = self.transform(image_b)
        
        cutoff = self.cutoffs[idx]

        #############################################################################
        #                             END OF YOUR CODE
        #############################################################################

        return image_a, image_b, cutoff


# TODO - 7
class HybridImageModel(nn.Module):
    def __init__(self):
        # Initializes an instance of the HybridImageModel class.
        super(HybridImageModel, self).__init__()

    def get_kernel(self, cutoff_standarddeviation: int) -> torch.Tensor:
        # Returns a Gaussian kernel using the specified cutoff standard deviation.
        #
        # PyTorch requires the kernel to be of a particular shape in order to apply
        # it to an image. Specifically, the kernel needs to be of shape (c, 1, k, k)
        # where c is the # channels in the image.
        #
        # Start by getting a 2D Gaussian kernel using your implementation from earlier,
        # which will be of shape (k, k). Then, let's say you have an RGB image, you
        # will need to turn this into a Tensor of shape (3, 1, k, k) by stacking the
        # Gaussian kernel 3 times.
        #
        # Args:
        #     cutoff_standarddeviation: int specifying the cutoff standard deviation
        # Returns:
        #     kernel: Tensor of shape (c, 1, k, k) where c is # channels
        #
        # HINTS:
        # - Since the # channels may differ across each image in the dataset, make
        #   sure you don't hardcode the dimensions you reshape the kernel to. There
        #   is a variable defined in this class to give you channel information.
        # - You can use torch.reshape() to change the dimensions of the tensor.
        # - You can use torch's repeat() to repeat a tensor along specified axes.

        kernel = torch.Tensor()

        #############################################################################
        #                             YOUR CODE BELOW
        #############################################################################
        #c = 3
        #k = int(4*cutoff_standarddeviation+1)
        #k = torch.as_tensor(k)
        #mu = torch.tensor(k/2)
        #mu = mu.type(torch.FloatTensor)
        #sigma = torch.as_tensor(cutoff_standarddeviation)
        #kernel = [torch.exp(-(i-mu)**2/(2*sigma**2)) for i in range(k)]
        #kernel_1D = torch.FloatTensor(kernel)
        #kernel_2D = torch.ger(kernel_1D, kernel_1D)
        #kernel_2D = kernel_2D/kernel_2D.sum()
        
        #kernel = kernel_2D.expand(3, *kernel_2D.size()).unsqueeze(1)
        #raise NotImplementedError --------
        
        kernel_temp = create_2d_gaussian_kernel(cutoff_standarddeviation)
        kernel_temp = torch.unsqueeze(kernel_temp, 0)
        kernel = torch.unsqueeze(kernel_temp, 0).repeat(self.n_channels, 1, 1, 1)

        #############################################################################
        #                             END OF YOUR CODE
        #############################################################################

        return kernel

    def low_pass(self, x, kernel):
        # Apply low pass filter to the input image.
        #
        # Args:
        #     x: Tensor of shape (b, c, m, n) where b is batch size
        #     kernel: low pass filter to be applied to the image
        # Returns:
        #     filtered_image: Tensor of shape (b, c, m, n)
        #
        # HINT:
        # - You should use the 2d convolution operator from torch.nn.functional.
        # - Make sure to pad the image appropriately (it's a parameter to the
        #   convolution function you should use here!).
        # - Pass self.n_channels as the value to the "groups" parameter of the
        #   convolution function. This represents the # of channels that the filter
        #   will be applied to.

        filtered_image = torch.Tensor()

        #############################################################################
        #                             YOUR CODE BELOW
        #############################################################################
        #N,C,H,W = x.shape
        #C,N,HH,WW = kernel.shape
        B,C,M,N = x.shape
        C,B,MM,NN = kernel.shape
        horizontal_pad = int(((MM-1)/2)+0.5)
        vertical_pad = int(((NN-1)/2)+0.5)
        filtered_image = F.conv2d(x, kernel, padding=(horizontal_pad,vertical_pad), groups = 3)

        #raise NotImplementedError

        #############################################################################
        #                             END OF YOUR CODE
        #############################################################################

        return filtered_image

    def forward(self, image1, image2, cutoff_standarddeviation):
        # Take two images and creates a hybrid image. Returns the low frequency
        # content of image1, the high frequency content of image 2, and the hybrid
        # image.
        #
        # Args:
        #     image1: Tensor of shape (b, m, n, c)
        #     image2: Tensor of shape (b, m, n, c)
        #     cutoff_standarddeviation: Tensor of shape (b)
        # Returns:
        #     low_frequencies: Tensor of shape (b, m, n, c)
        #     high_frequencies: Tensor of shape (b, m, n, c)
        #     hybrid_image: Tensor of shape (b, m, n, c)
        #
        # HINTS:
        # - You will use the get_kernel() function and your low_pass() function in
        #   this function.
        # - Don't forget to make sure to clip the pixel values >=0 and <=1. You can
        #   use torch.clamp().
        # - If you want to use images with different dimensions, you should resize
        #   them in the HybridImageDataset class using torchvision.transforms.

        self.n_channels = image1.shape[1]

        low_frequencies = torch.Tensor()
        high_frequencies = torch.Tensor()
        hybrid_image = torch.Tensor()

        #############################################################################
        #                             YOUR CODE BELOW
        #############################################################################
        kernel = self.get_kernel(cutoff_standarddeviation)
        low_frequencies = self.low_pass(image1, kernel)
        #low_pass2 = self.low_pass(image2, kernel)
        high_frequencies = image2 - self.low_pass(image2,kernel)
        hybrid_image = low_frequencies + high_frequencies
        
        hybrid_image = torch.clamp(hybrid_image,0.0,1.0)
        #raise NotImplementedError

        #############################################################################
        #                             END OF YOUR CODE
        #############################################################################

        return low_frequencies, high_frequencies, hybrid_image


# TODO - 8
def my_median_filter(image: torch.FloatTensor, filter_size: Union[tuple, int]) -> torch.FloatTensor:
    """
    Apply a median filter to an image. Return the filtered image.
    Args
    - image: Torch tensor of shape (m, n, 1) or Torch tensor of shape (m, n).
    - filter: Torch tensor of shape (k, j). If an integer is passed then all dimensions
              are considered of the same size. Input will always have odd size.
    Returns
    - filtered_image: Torch tensor of shape (m, n, 1)
    HINTS:
    - You may not use any libraries that do the work for you. Using torch to work
     with matrices is fine and encouraged. Using OpenCV/scipy or similar to do the
     filtering for you is not allowed.
    - I encourage you to try implementing this naively first, just be aware that
     it may take a long time to run. You will need to get a function
     that takes a reasonable amount of time to run so that the TAs can verify
     your code works.
    - Useful functions: torch.median and torch.nn.functional.pad
    """
    if len(image.size()) == 3:
        assert image.size()[2] == 1

    if isinstance(filter_size, int):
        filter_size = (filter_size, filter_size)
    assert filter_size[0] % 2 == 1
    assert filter_size[1] % 2 == 1

    filtered_image = torch.Tensor()

    ############################################################################
    # TODO: YOUR CODE HERE
    ############################################################################
    squeezed_image = torch.squeeze(image)
    filtered_image = torch.zeros_like(squeezed_image)
    image_filter = filter_size
    #print(filter_size)
    row_pad = ((image_filter[0]-1)//2)
    col_pad = ((image_filter[1]-1)//2)
   
    image_pad = F.pad(image,pad=(col_pad,col_pad,row_pad,row_pad),mode = 'constant', value = 0)
    #print(image_pad.shape)
    for i in range(filtered_image.shape[0]):
        for j in range(filtered_image.shape[1]):
            filtered_image[i,j] = (torch.median(image_pad[i:i+image_filter[0],j:j+image_filter[1]]))
    
    filtered_image = torch.unsqueeze(filtered_image,dim=0)
    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################
    return filtered_image


#############################################################################
# Extra credit opportunity (for UNDERGRAD) below
#
# Note: This part is REQUIRED for GRAD students
#############################################################################

# Matrix multiplication helper
def complex_multiply_real(m1, m2):
    # Take the one complex tensor matrix and a real matrix, and do matrix multiplication
    # Args:
    #     m1: the Tensor matrix (m,n,2) which represents complex number;
    #         E.g., the real part is t1[:,:,0], the imaginary part is t1[:,:,1]
    #     m2: the real matrix (m,n)
    # Returns
    #     U: matrix multiplication result in the same form as input m1

    real1 = m1[:, :, 0]
    imag1 = m1[:, :, 1]
    real2 = m2
    imag2 = torch.zeros(real2.shape)
    return torch.stack([torch.matmul(real1, real2) - torch.matmul(imag1, imag2),
                        torch.matmul(real1, imag2) + torch.matmul(imag1, real2)], dim=-1)


# Matrix multiplication helper
def complex_multiply_complex(m1, m2):
    # Take the two complex tensor matrix and do matrix multiplication
    # Args:
    #     t1, t2: the Tensor matrix (m,n,2) which represents complex number;
    #             E.g., the real part is t1[:,:,0], the imaginary part is t1[:,:,1]
    # Returns
    #     U: matrix multiplication result in the same form as input

    real1 = m1[:, :, 0]
    imag1 = m1[:, :, 1]
    real2 = m2[:, :, 0]
    imag2 = m2[:, :, 1]
    return torch.stack([torch.matmul(real1, real2) - torch.matmul(imag1, imag2),
                        torch.matmul(real1, imag2) + torch.matmul(imag1, real2)], dim=-1)


# TODO - 9
def dft_matrix(N):
    # Take the square matrix dimension as input, generate the DFT matrix correspondingly
    #
    # Args:
    #     N: the DFT matrix dimension
    # Returns:
    #     U: the generated DFT matrix (torch.Tensor) of size (N,N,2);
    #         the real part is represented by U[:,:,0],
    #         and the complex part is represented by U[:,:,1]

    U = torch.Tensor()

    torch.pi = torch.acos(torch.zeros(1)).item() * \
        2  # which is 3.1415927410125732

    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################
    U = torch.zeros(N,N,2)
    for i in range(N):
        for j in range(N):
            U[i,j,0] = (1/N)*torch.cos(torch.tensor(2*torch.pi*i*j)/N)
            U[i,j,1] = (-1/N)*torch.sin(torch.tensor(2*torch.pi*i*j)/N)

    #raise NotImplementedError

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    return U


# TODO - 10
def my_dft(img):
    # Take a square image as input, perform Discrete Fourier Transform for the image matrix
    # This function is expected to behave similar as torch.rfft(x,2,onesided=False) except a scale parameter
    #
    # Args:
    #     img: a 2D grayscale image (torch.Tensor) whose width equals height, size: (N,N)
    # Returns:
    #     dft: the DFT results of img; the size should be (N,N,2),
    #          where the real part is dft[:,:,0], while the imag part is dft[:,:,1]
    #
    # HINT:
    # - We provide two function to do complex matrix multiplication:
    #   complex_multiply_real and complex_multiply_complex

    dft = torch.Tensor()

    assert img.shape[0] == img.shape[1], "Input image should be a square matrix"

    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################
    N = img.size()[0]
    U = dft_matrix(N)
    
    holdon = complex_multiply_real(U,img)
    dft = complex_multiply_complex(holdon,U)

    #raise NotImplementedError

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    return dft


# TODO - 11
def dft_filter(img):
    # Take a square image as input, performs a low-pass filter and return the filtered image
    #
    # Args
    # - img: a 2D grayscale image whose width equals height, size: (N,N)
    # Returns
    # - img_back: the filtered image whose size is also (N,N)
    #
    # HINTS:
    # - You will need your implemented DFT filter for this function
    # - We don't care how much frequency you want to retain, if only it returns reasonable results
    # - Since you already implemented DFT part, you're allowed to use the torch.ifft in this part for convenience, though not necessary

    img_back = torch.Tensor()

    #############################################################################
    #                             YOUR CODE BELOW
    #############################################################################
    dft = my_dft(img)
    N = img.size()[0]
    masking = torch.zeros(N,N,2)
    masking[0, 0, :] = 1
    masking[0, N-1, :] = 1
    masking[N-1, 0, :] = 1
    masking[N-1, N-1, :] = 1
    holdon = dft*masking
    ifft_computed = torch.ifft(holdon,2)
    img_back = ifft_computed[:, :, 0]

    #raise NotImplementedError

    #############################################################################
    #                             END OF YOUR CODE
    #############################################################################

    return img_back
