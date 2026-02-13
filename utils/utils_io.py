from PIL import Image
import OpenEXR
import Imath
import numpy as np
import torch
from torchvision import transforms

pascal_channel_id = {
    'aeroplane':    1,
    'bicycle':      2,
    'bird':         3,
    'boat':         4,
    'bottle':       5,
    'bus':          6,
    'car':          7,
    'cat':          8,
    'chair':        9,
    'cow':         10,
    'dog':         12,
    'horse':       13,
    'motorbike':   14,
    'person':      15,
    'pottedplant': 16,
    'sheep':       17,
    'train':       19,
    'tvmonitor':   20,
    }


def read_exr(file_path):
    """Reads an EXR file and combines its channels into a single NumPy array.

    Args:
        file_path (str): Path to the EXR file.

    Returns:
        np.ndarray: Combined image array with shape (height, width, channels).
        list: List of channel names that were combined into the array.
    """
    # Open the EXR file
    exr_file = OpenEXR.InputFile(file_path)

    # Get the header to extract the data window and channel information
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    # Retrieve channels available in the header
    channel_list = list(header['channels'].keys())

    # Create a pixel type descriptor for reading in floats (commonly used)
    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    # Read each channel data
    channel_data = {c: np.frombuffer(exr_file.channel(c, pt), dtype=np.float32) for c in channel_list}

    # Reshape the channel data to match the original image dimensions
    channel_data = {c: data.reshape(height, width) for c, data in channel_data.items()}

    # Check for NaN values in each channel
    for c, data in channel_data.items():
        if np.isnan(data).any():
            print(f"Warning: NaN values found in channel {c}")

    # Assuming all channels have identical dimensions, stack them to form a multi-channel array
    combined_image = np.stack([channel_data[c] for c in sorted(channel_list)], axis=-1)

    return combined_image, channel_list


def get_pascal_mask(mask_path):

    cat = mask_path.split('/')[-2]

    mask = Image.open(mask_path).convert('L')
    mask = transforms.functional.to_tensor(mask)
    mask = (mask == ((pascal_channel_id[cat]) / 255.0)).float()
    mask = mask.view(1,1,mask.size(1),mask.size(2))

    return mask

def get_owl_mask(mask_path):

    mask = torch.from_numpy(np.load(mask_path))
    mask = mask.view(1,1,mask.size(0),mask.size(1))

    return mask