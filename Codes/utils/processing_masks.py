import torch
import numpy as np
import cv2

def region_growing(img, seed):
    # Initialize the mask with all ones (background)
    mask = np.zeros_like(img)

    # Set the seed point as the first pixel to be added to the region
    region_pixels = [seed]

    # Loop until no more pixels can be added to the region
    while region_pixels:

        # Get the next pixel to consider from the region
        pixel = region_pixels.pop(0)

        # Check if the pixel is already in the region
        if mask[pixel] == 1:
            continue

        # Get the value of the seed pixel
        seed_value = img[seed]

        # Add the current pixel to the region if it has the same value as the seed pixel
        if img[pixel] == seed_value:
            mask[pixel] = 1  # Mark the pixel as part of the region
            # Get the neighbors of the pixel
            neighbors = get_neighbors(pixel, img.shape)
            # Add the neighbors with the same value to the region
            for neighbor in neighbors:
                if mask[neighbor] == 0 and img[neighbor] == seed_value:
                    region_pixels.append(neighbor)

    return mask
def get_neighbors(pixel, shape):
    neighbors = []
    row, col = pixel
    if row > 0:
        neighbors.append((row - 1, col))
    if row < shape[0] - 1:
        neighbors.append((row + 1, col))
    if col > 0:
        neighbors.append((row, col - 1))
    if col < shape[1] - 1:
        neighbors.append((row, col + 1))
    return neighbors

def pred_to_segment_3ch(subject, nChannel, label_colours, slices):
    w = 500
    volume = torch.empty((slices, w, w, 3), dtype=torch.long)
    for s in range(slices):
        output = subject[:, :, :, s]
        output = output.permute(2, 1, 0).contiguous().view(-1, nChannel*2)
        ignore, target = torch.max(output, 1)
        im_target = target.data.cpu().numpy()
        im_target_rgb = np.array([label_colours[c % nChannel] for c in im_target])
        im_target_rgb = im_target_rgb.reshape((w, w, 3)).astype(np.uint8)
        volume[s] = torch.from_numpy(im_target_rgb)
    return volume

def prediction_mask_computation(sbj, mask, min_thresh, max_thresh):
    mask = mask.astype(np.uint8)
    num_labels, labeled_mask = cv2.connectedComponents(mask, connectivity=8)
    tumor_labels = [np.where(labeled_mask == label, label, 0) for label in range(1, num_labels)]
    resulting_masks = []
    if len(tumor_labels)==0:
        pre_mask = mask[:,:,0]
        resulting_masks.append(pre_mask)
    else:
        for tumor in range(len(tumor_labels)):
            tumor_mask = tumor_labels[tumor]
            tumor_mask = np.expand_dims(tumor_mask, axis=-1)
            tumor_mask[tumor_mask < 0.5] = 0
            tumor_mask[tumor_mask >= 0.5] = 1

            tumor_area = np.sum(tumor_mask == 1)

            # 3. Multiply mask by segmentation to get the tumor region
            mask_3_channel = np.repeat(tumor_mask, 3, axis=-1)
            white_image = np.zeros_like(sbj) * 255
            result = np.where(mask_3_channel == 1, sbj, white_image)
            result_8u = cv2.convertScaleAbs(result)
            gray_prediction = cv2.cvtColor(result_8u, cv2.COLOR_BGR2GRAY)
            pre_mask = np.zeros_like(gray_prediction, dtype=np.uint8)

            # 4. Pick clusters inside the tumor
            unique_values, counts = np.unique(gray_prediction, return_counts=True)
            unique_values = unique_values[1:] # Remove value 0
            counts = counts[1:] # Remove value 0

            # 5. Keep cluster that occupy at least 70% of tumor area
            cluster = []
            for i in range(len(counts)):
                if counts[i]/tumor_area >= min_thresh :
                    cluster.append(unique_values[i])

            # Set seed for region growing
            seed_mask = np.zeros_like(gray_prediction, dtype=np.uint8)
            coordinates = []
            for cluster_label in cluster:
                seed_region = np.where(gray_prediction == cluster_label)
                seed_mask[seed_region] = 1
                seed_indices = np.where(seed_mask == 1)
                num_seed_indices = len(seed_indices[0])
                random_index = random.randint(0, num_seed_indices - 1)
                random_seed_coordinate = (seed_indices[0][random_index], seed_indices[1][random_index])
                coordinates.append(random_seed_coordinate)

            if coordinates is not None:
                list_masks = []
                for coord in range(len(coordinates)):
                    sbj_8u = cv2.convertScaleAbs(sbj) 
                    gray_prediction2 = cv2.cvtColor(sbj_8u, cv2.COLOR_BGR2GRAY)
                    coord_mask = region_growing(gray_prediction2, coordinates[coord])
                    mask_tumor_area = np.sum(coord_mask == 1)
                    #print(f'cluster area {coord}: {mask_tumor_area}')
                    if mask_tumor_area > max_thresh*tumor_area:
                        coord_mask = np.zeros_like(gray_prediction, dtype=np.uint8)
                        #print('discarded')
                    list_masks.append(coord_mask)
                #print(f'len list_masks: {len(list_masks)}')
                pre_mask = np.sum(np.array(list_masks), axis=0)



            resulting_masks.append(pre_mask)

    final_mask = np.sum(np.array(resulting_masks), axis=0)
    return final_mask