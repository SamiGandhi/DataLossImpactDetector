import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# the seed function is used to generate the same random series each time the code runs
np.random.seed(123)

# setup and initialize feature detectors
sift = cv2.SIFT_create()
kaze = cv2.KAZE_create()
akaze = cv2.AKAZE_create()
orb = cv2.ORB_create()
fast = cv2.FastFeatureDetector_create()

# Create the feature matchers
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
flann = cv2.FlannBasedMatcher()


def get_images_filenames(images_directory):
    file_names = []
    for file in os.listdir(images_directory):
        file_names.append(os.path.join(images_directory, file))
    return file_names


def simulate_noises(image):
    generated_images = []
    # image dimensions
    height, width, _ = image.shape
    # maximum size of the black box to 10% of the image dimensions
    max_size = int(min(height, width) * 0.1)
    # initial size of the black box to 1 pixel
    box_size = 1
    # Set the initial noise level to 0
    noise_level = 0
    for i in range(10):
        # generate the number of black boxes to be added
        num_boxes = np.random.randint(1, 5)
        # generate the coordinates of the black boxes
        box_coords = [(np.random.randint(0, height - 1), np.random.randint(0, width - 1)) for _ in range(num_boxes)]
        # increase the size of the black boxes
        box_size += int(max_size / 10)
        # Increase the noise level
        noise_level += 10
        # Add the black boxes to the image
        for x, y in box_coords:
            x1, y1 = max(x - box_size, 0), max(y - box_size, 0)
            x2, y2 = min(x + box_size, height - 1), min(y + box_size, width - 1)
            image[x1:x2, y1:y2, :] = (0, 0, 0)
        # gaussian noise to simulate transmission noise
        image = cv2.GaussianBlur(image, (3, 3), noise_level)
        element = "noise_level" + str(i), image
        generated_images.append(element)
    # the generated image is an array that contains the entered image with 10 levels of data loss and noises
    return generated_images


def get_ref_images(ref_images_directory):
    result = []
    i = 0
    for ref_image_path in ref_images_directory:
        img = cv2.imread(ref_image_path)
        element = "Ref_Image" + str(i), img
        result.append(element)
    return result


def get_test_images(ref_images_directory):
    result = []
    i = 0
    for ref_image_path in ref_images_directory:
        img = cv2.imread(ref_image_path)
        element = "Test_Image" + str(i), img
        result.append(element)
    return result


def detect_and_compute_key_points(image_list):
    result = []
    for image in image_list:
        # extract the image and its names
        name, img = image
        _kp_sift, _des_sift = sift.detectAndCompute(img, None)
        _kp_kaze, _des_kaze = kaze.detectAndCompute(img, None)
        _kp_akaze, _des_akaze = akaze.detectAndCompute(img, None)
        _kp_orb, _des_orb = orb.detectAndCompute(img, None)
        element = name, img, _kp_sift, _des_sift, _kp_kaze, _des_kaze, _kp_akaze, _des_akaze, _kp_orb, _des_orb
        result.append(element)
    return result


def match_features(ref_images_kp_desc_array, noises_levels_kp_desc_array):
    result = []
    for ref_images_kp_desc in ref_images_kp_desc_array:
        # extract the descriptors
        name, ref_image, ref_kp_sift, ref_descr_sift, ref_kp_kaze, ref_descr_kaze, \
        ref_kp_akaze, ref_descr_akaze, ref_kp_orb, ref_descr_orb = ref_images_kp_desc

        for noise_image_kp_descr in noises_levels_kp_desc_array:
            # extract the descriptors
            name, noise_image, noise_kp_sift, noise_descr_sift, noise_kp_kaze, noise_descr_kaze, \
            noise_kp_akaze, noise_descr_akaze, noise_kp_orb, noise_descr_orb = noise_image_kp_descr
            matches_sift = bf.match(ref_descr_sift, noise_descr_sift)
            matches_kaze = bf.match(ref_descr_kaze, noise_descr_kaze)
            matches_akaze = bf.match(ref_descr_akaze, noise_descr_akaze)
            matches_orb = bf.match(ref_descr_orb, noise_descr_orb)
            # add the matches into the result array
            element = ref_images_kp_desc, noise_image_kp_descr, matches_sift, matches_kaze, matches_akaze, matches_orb
            result.append(element)

    return result


def positive_matches(matches_list):
    result = []
    for matches in matches_list:
        # extract the previous results
        ref_images_kp_desc, noise_image_kp_descr, matches_sift, matches_kaze, matches_akaze, matches_orb = matches
        num_positive_matches_sift = 0
        for match in matches_sift:
            if match.distance < 0.75 * matches_sift[0].distance:
                num_positive_matches_sift += 1

        num_positive_matches_kaze = 0
        for match in matches_kaze:
            if match.distance < 0.75 * matches_kaze[0].distance:
                num_positive_matches_kaze += 1

        num_positive_matches_akaze = 0
        for match in matches_akaze:
            if match.distance < 0.75 * matches_akaze[0].distance:
                num_positive_matches_akaze += 1

        num_positive_matches_orb = 0
        for match in matches_orb:
            if match.distance < 0.75 * matches_orb[0].distance:
                num_positive_matches_orb += 1
        element = matches, num_positive_matches_sift, num_positive_matches_kaze, num_positive_matches_akaze, num_positive_matches_orb
        result.append(element)
    return result


# this method here uses cv2.cv2.findHomography to detect how much object detected

def counting_object(ref_kp, test_kp, good_matches):
    ref_pts = np.float32([ref_kp[m[0].queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    test_pts = np.float32([test_kp[m[0].trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(ref_pts, test_pts, cv2.RANSAC, 5.0)
    num_positive_matches = mask.sum()
    return num_positive_matches


def plot_results(positives):
    n = 0
    i = 0
    x = []
    y_sift_features = []
    y_kaze_features = []
    y_akaze_features = []
    y_orb_features = []

    y_sift_all_matches = []
    y_sift_positive_matches = []
    y_kaze_all_matches = []
    y_kaze_positive_matches = []
    y_akze_all_matches = []
    y_akaze_positive_matches = []
    y_orb_all_matches = []
    y_orb_positive_matches = []

    for positive in positives:
        matches, num_positive_matches_sift, num_positive_matches_kaze, num_positive_matches_akaze, num_positive_matches_orb = positive
        ref_images_kp_desc, noise_image_kp_descr, matches_sift, matches_kaze, matches_akaze, matches_orb = matches
        name, noise_image, noise_kp_sift, noise_descr_sift, noise_kp_kaze, noise_descr_kaze, noise_kp_akaze, noise_descr_akaze, noise_kp_orb, noise_descr_orb = noise_image_kp_descr = noise_image_kp_descr
        print("Noise Image: -> " + str(name))
        x.append(i)
        print("    SIFT FEATURES: -> " + str(len(noise_kp_sift)))
        y_sift_features.append(len(noise_kp_sift))
        print("    KAZE FEATURES: -> " + str(len(noise_kp_kaze)))
        y_kaze_features.append(len(noise_kp_kaze))
        print("    AKAZE FEATURES: -> " + str(len(noise_kp_akaze)))
        y_akaze_features.append(len(noise_kp_akaze))
        print("    ORB FEATURES: -> " + str(len(noise_kp_orb)))
        y_orb_features.append(len(noise_kp_orb))
        print("    All matches SIFT : -> " + str(len(matches_sift)))
        y_sift_all_matches.append(len(matches_sift))
        print("    Positive matches SIFT: -> " + str(num_positive_matches_sift))
        y_sift_positive_matches.append(num_positive_matches_sift)
        print("    All matches KAZE : -> " + str(len(matches_kaze)))
        y_kaze_all_matches.append(len(matches_kaze))
        print("    Positive matches KAZE: -> " + str(num_positive_matches_kaze))
        y_kaze_positive_matches.append(num_positive_matches_kaze)
        print("    All matches AKAZE : -> " + str(len(matches_akaze)))
        y_akze_all_matches.append(len(matches_akaze))
        print("    Positive matches AKAZE: -> " + str(num_positive_matches_akaze))
        y_akaze_positive_matches.append(num_positive_matches_akaze)
        print("    All matches ORB : -> " + str(len(matches_orb)))
        y_orb_all_matches.append(len(matches_orb))
        print("    Positive matches ORB: -> " + str(num_positive_matches_orb))
        y_orb_positive_matches.append(num_positive_matches_orb)
        if i == 9:
            fig, ax = plt.subplots()
            ax.plot(x, y_sift_features, label='sift features')
            ax.plot(x, y_kaze_features, label='kaze features')
            ax.plot(x, y_akaze_features, label='akaze features')
            ax.plot(x, y_orb_features, label='orb features')
            ax.legend()
            # Set labels for the x and y axes
            ax.set_xlabel('Data loss augmentation')
            ax.set_ylabel('Detected features')
            # Set a title for the plot
            ax.set_title('Detected features over data loss')
            plt.savefig('features_image' + str(n) + '.png')
            plt.clf()
            fig, ax = plt.subplots()
            ax.plot(x, y_sift_all_matches, label='sift all matches')
            ax.plot(x, y_kaze_all_matches, label='kaze all matches')
            ax.plot(x, y_akze_all_matches, label='akaze all matches')
            ax.plot(x, y_orb_all_matches, label='orb all matches')
            ax.plot(x, y_sift_positive_matches, label='sift positive matches')
            ax.plot(x, y_kaze_positive_matches, label='kaze positive matches')
            ax.plot(x, y_akaze_positive_matches, label='akaze positive matches')
            ax.plot(x, y_orb_positive_matches, label='orb positive matches')
            ax.legend()
            # Set labels for the x and y axes
            ax.set_xlabel('Data loss augmentation')
            ax.set_ylabel('Matches')
            # Set a title for the plot
            ax.set_title('Matches over data loss')
            plt.savefig('matches_image' + str(n) + '.png')
            plt.clf()
            n = n + 1
            i = 0
            x = []
            y_sift_features = []
            y_kaze_features = []
            y_akaze_features = []
            y_orb_features = []

            y_sift_all_matches = []
            y_sift_positive_matches = []
            y_kaze_all_matches = []
            y_kaze_positive_matches = []
            y_akze_all_matches = []
            y_akaze_positive_matches = []
            y_orb_all_matches = []
            y_orb_positive_matches = []
        else:
            i = i + 1

test_images_file_names = get_images_filenames('test_images')
ref_images_file_names = get_images_filenames('ref_images')
test_images_ = get_test_images(test_images_file_names)
ref_images_ = get_ref_images(ref_images_file_names)
noise_images = []
for name, image in test_images_:
    noise_images_ = simulate_noises(image)
    for noise in noise_images_:
        noise_images.append(noise)

ref_images_kp_descr = detect_and_compute_key_points(ref_images_)
noise_images_kp_descr = detect_and_compute_key_points(noise_images)
matching = match_features(ref_images_kp_descr, noise_images_kp_descr)
positives = positive_matches(matching)
plot_results(positives)

