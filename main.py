# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import read

# align the longitudinal axis with the Z-axis
def AlignZ_Rot_mat (xx):
    # creates rotation matrix
    theta = xx[0]
    phi = xx[1]
    temp = np.array([
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), -np.cos(theta)],
        [-np.sin(phi), np.cos(phi), 0],
        [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), np.sin(theta)]
    ])
    return temp
# Press the green button in the gutter to run the script.

if __name__ == '__main__':
    bvals = read.readFile('dataset/bvals')
    bvecs = read.readFile('dataset/bvecs')

    bvecs_jones = read.readFile('dataset/bvecs_jones.txt')
    bvecs_jones = bvecs_jones[7:71]
    bvecs_64_prime = np.vstack((bvecs_jones, -1*bvecs_jones))

    # bvecs_64_prime_last_col = np.matmul(bvecs_64_prime, np.array([0,0,1]))

    # three_index = bvecs_64_prime.view('i8,i8,i8')
    three_index = bvecs_64_prime[bvecs_64_prime[:, -1].argsort()][-1: -4 : -1]

    # acos(bvecs_64.prime[three_index,][1,] % * % bvecs_64.prime[three_index,][2,]) / pi * 180
    # acos(bvecs_64.prime[three_index,][1,] % * % bvecs_64.prime[three_index,][3,]) / pi * 180
    # acos(bvecs_64.prime[three_index,][2,] % * % bvecs_64.prime[three_index,][3,]) / pi * 180
    cos1 = np.arccos(np.matmul(three_index[0], three_index[1])) / np.pi * 180
    cos2 = np.arccos(np.matmul(three_index[0], three_index[2])) / np.pi * 180
    cos3 = np.arccos(np.matmul(three_index[1], three_index[2])) / np.pi * 180

    ######!!!!!!!!!!!!!!!!##############
    middle_direction = np.array([0.17995, 0.0406928, 0.982834])
    ########!!!!!!!!!!!!!!##############

    #######!!!!!!!!!!##########
    # I don't know how to transform coordinate
    # middle.direction.theta < -cart2sph(as.vector(middle.direction))[2]
    middle_direction_theta = 1.38523973877922
    # middle.direction.phi < -cart2sph(as.vector(middle.direction))[1]
    middle_direction_phi = 0.222393467076206

    # rotate all gradient directions, as we rotate the middle direction to Z-axis
    bvecs_64 = np.matmul(AlignZ_Rot_mat(np.array([middle_direction_theta, middle_direction_phi]),),
                         bvecs_64_prime.transpose()).transpose()
    for i in range(128):
        bvecs_64[i] = bvecs_64[i] / np.sqrt(np.sum(bvecs_64[0]**2))

    # create the hypothetical gradients amid the true gradients

    # reference.dir.theta < -cart2sph(as.vector(reference.dir))[2]
    reference_dir_theta = 1.36152096855267
    # reference.dir.phi < -cart2sph(as.vector(reference.dir))[1]
    reference_dir_phi = 0.655321011430509
    bvecs_64_hypo = np.matmul(AlignZ_Rot_mat(np.array([reference_dir_theta, reference_dir_phi]), ),
                         bvecs_64.transpose()).transpose()
    print(bvecs_64_hypo)