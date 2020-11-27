import numpy as np
from numpy import cos, sin, pi, exp, sqrt, abs, round, arccos, arcsin
import read
from scipy.stats import norm, vonmises
import matplotlib.pyplot as plt


def cart2sph(cart):
    x, y, z = cart[0], cart[1], cart[2]
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    r = np.sqrt(x**2 + y**2 + z**2)
    return [azimuth, elevation, r]


def sph2cart(azimuth, elevation, r):
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z


# align the longitudinal axis with the Z-axis
def AlignZ_Rot_mat (xx):
    # creates rotation matrix
    theta = xx[0]
    phi = xx[1]
    temp = np.array([
        [sin(theta) * cos(phi), sin(theta) * sin(phi), -cos(theta)],
        [-sin(phi), cos(phi), 0],
        [cos(theta) * cos(phi), cos(theta) * sin(phi), sin(theta)]
    ])
    return temp


# function used in function "TwoFiber_Behrens_simplified_non_informative_simulation_sim",
# to calculate the posterior log-likelihood in MCMC sampling
def log_post_2(params, y_value, b, est_sum_f, est_d):
    # log posterior for the model
    # (1)
    f1 = est_sum_f


def rotation_matrix(theta, phi):
    # create rotation matrix
    ########## !!!!!!!!!!!! ###########
    # here has different with R code and paper, I take the R code version
    return np.array([[cos(theta) * cos(phi), -1 * sin(phi), sin(theta) * cos(phi)],
                    [cos(theta) * sin(phi), cos(phi), sin(theta) * sin(phi)],
                    [sin(theta), 0, -1 * cos(theta)]])


def TwoFiber_Behrens_simplified_non_informative_simulation_sim(y_value, b, est_sum_f, est_d):
    output = []
    num_scan = 100000 # number of MCMC scans
    thin = 10 #thin interval

    est_1 = []
    log_post_likelihood = []

    ## prior specification for sigma^2: Gibbs sampling in InverseGamma fucntion parameterized by shape.sigma, scale.sigma
    # set mean of error E(error)=0.005=1/theta*(k-1); k:shape, theta: scale
    # prior parameters for sigma^2
    shape_sigma = 101
    scale_sigma = 1400

    # starting values
    new_1 = [0, pi, pi]  # modified
    prop_1 = new_1
    sigma_sq_1 = 14  # intial value for noise (sigma.square)

    # starting log standard deviations for the component-wise proposal distributions
    log_s1 = [-1, -1, -1]  # 3: number of params (except sigma.sq)

    # starting acceptance rates
    ar_1 = [0, 0, 0]

    # starting acceptance counts
    accept_count_1 = [0, 0, 0]

    log_s = log_s1

    counts = []

    for i in range(num_scan):
        if i % 1000 == 0:
            print(i + "\n")
        # get the batch number
        batch_num = i / 50 + 1 # batch size is 50
        # get the iteration number inside a batch
        iter_batch = i % 50
        delta = min(0.01, 1/sqrt(batch_num))

        # first iteration in a batch of 50,
        # so update acceptance rates, reset counters and update proposal variances
        if (iter_batch == 0):
            # update the acceptance rates
            ar_1 = accept_count_1/50
            counts.add(ar_1)

            # reset the acceptance counters
            accept_count_1 = [0, 0, 0]


def erf(x):
    return 2 * norm.cdf(x * sqrt(2)) - 1


# simulation function of simplified Behrens model
# input:
# vol.fracs: the vector of fiber volume fraction
# kappa: parameter of generating von-mises weights. Larger kappa indicates smaller weights borrowing from neighboring gradients
# phis: the vector of angle phi of fiber
# b: diffusion weighting factor
# d: diffusivity
# noise.level: assummed noisy level (e.g. 5% of baseline intensity S0)
def SmootherEstimator_sim(vol_fracs, phis, kappa=None,  b=1500, d=1/1500, noise_level=0.05, S0):
    # Nsim=1000; vol.fracs=c(0.3,0.3); phis=c(60, 120); kappa=10
    # Nsim: number of iterations
    # vol.fracs (vector): volume fraction for 2 fibers
    # phis (vector): phi angle for 2 fibers
    MSE_noise = MSE_smooth = smooth_max_bias = noise_max_bias = smooth_angle_biases = noise_angle_biases = None

    # if kappa has been assigned a value (e.g. not equal to 5 ),
    # the density function of Von-Mises should be changed
    if kappa != None:
        for i in range(scale.shape[0]):
            weight_vonmises[i, :] = vonmises.pdf(nearest_angle[i, :], kappa=kappa)
            scale[i] = np.sum(weight_vonmises[i, :])
            inv_scale[i] = 1 / scale[i]

    # 64-point observed intensity data: non-noisy, noisy
    # vol.fracs=c(0.2,0.7); phis=c(30,120); kappa=5
    # raw signal data across the true gradients (e.g. 64 directions)
    seed = np.random.choice(1000000, 1)[0]
    np.random.seed(seed)

    signal_data = BehrensSim(S0=400, f=vol_fracs, bval=np.full((bvecs_64.shape[0]), b), bvecs=bvecs_64, d=d, theta=np.array([0, 0]), phi=phis)
    # noise = np.random.normal(0, S0 * noise_level, int(bvecs_64.shape[0] / 2))
    noise = norm.rvs(scale = S0 * noise_level, size=int(bvecs_64.shape[0] / 2) )

# Simulate mean S_i (non-noisy data) based on Behrens model
def BehrensSim (S0, f, bval, bvecs, d, theta, phi):
    if len(f) != len(theta) or len(f) != len(phi):
        print("Warning: parameters involving the number of fibers do not have the same length!!")
        return
    A_mat = np.array([[1,0,0],[0,0,0],[0,0,0]]) # A matrix
    ball_comp = (1 - np.sum(f)) * np.exp(-1 * bval * d) # ball component
    stick_comp = np.zeros(len(bval))
    for k in range(len(f)): # compute stick component
        theta_k = theta[k] * pi /180
        phi_k = phi[k] * pi /180
        R = rotation_matrix(theta=theta_k, phi=phi_k)
        stick_comp += f[k] * exp(-1 * bval * d * np.diag(bvecs @ R @ A_mat @ R.transpose() @ bvecs.transpose()))
    Si = S0 * (ball_comp + stick_comp)
    return Si



if __name__ == '__main__':
    # bvals = read.readFile('dataset/bvals')
    # bvecs = read.readFile('dataset/bvecs')

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
    # what is this??
    middle_direction = np.array([0.17995, 0.0406928, 0.982834])
    ########!!!!!!!!!!!!!!##############

    middle_direction_sph = cart2sph(middle_direction)
    middle_direction_theta = middle_direction_sph[1]
    middle_direction_phi = middle_direction_sph[0]

    # rotate all gradient directions, as we rotate the middle direction to Z-axis
    bvecs_64 = np.matmul(AlignZ_Rot_mat(np.array([middle_direction_theta, middle_direction_phi]),),
                         bvecs_64_prime.transpose()).transpose()
    for i in range(bvecs_64.shape[0]):
        bvecs_64[i] = bvecs_64[i] / np.sqrt(np.sum(bvecs_64[0]**2))

    # create the hypothetical gradients amid the true gradients
    reference_dir = bvecs_64[np.argmax(bvecs_64[:, 2])]
    reference_dir_sph = cart2sph(reference_dir)
    reference_dir_theta = reference_dir_sph[1]
    reference_dir_phi = reference_dir_sph[0]
    bvecs_64_hypo = np.matmul(AlignZ_Rot_mat(np.array([reference_dir_theta, reference_dir_phi]), ),
                         bvecs_64.transpose()).transpose()
    # print(arccos(max(bvecs_64[:, 2])) / pi * 180)
    # print(bvecs_64_hypo);
    # print(np.sort(arccos(bvecs_64_hypo[:, 2]) / pi * 180))

    bvecs_64_expand = np.vstack((bvecs_64, bvecs_64_hypo))

    # get the coordinates of 64 gradients, applied in the simplified model formula
    r_x = bvecs_64[:, 0]
    r_y = bvecs_64[:, 1]
    length_r_xy = r_x ** 2 + r_y ** 2
    phi_r = np.arctan2(r_y, r_x)

    # calculate all weight based on Von-mises distribution and angular separation
    # calculate the dot product of two directions
    #       (i-th out of 128 hypothetical directions and j-th out 64 gradients),
    # in order to obtain the pairwise angular separation between directions
    cos_angle = np.zeros((bvecs_64_expand.shape[0], bvecs_64.shape[0]))
    for i in range(bvecs_64_expand.shape[0]):
        for j in range(bvecs_64.shape[0]):
            cos_angle[i][j] = np.matmul(bvecs_64_expand[i, :], bvecs_64[j, :])
            if cos_angle[i, j] > 1:
                # some bvecs_jones directions dot product>1
                # cos.angle[i,j]<- round(cos.angle[i,j])
                cos_angle[i, j] = 1
            if cos_angle[i, j] < -1:
                cos_angle[i, j] = -1

    angle_from_most_neighb = np.array([])
    for i in range(cos_angle.shape[1]):
        angle_from_most_neighb = np.append(angle_from_most_neighb, np.sort(arccos(cos_angle[:, i]))[1] / pi * 180) # can be improved
    # summary(angle.from .MostNeighb)
    # sort(acos(cos.angle[32,]) / pi * 180)

    # calculate i-th most neighboring angular separation of each gradient
    mean_angular_sep = np.array([])
    sd_angular_sep = np.array([])
    # for (i in 2: 127) {
    #     mean.angular.sep < - c(mean.angular.sep, mean(acos(apply(cos.angle, 1, function(x) x[order(x, decreasing = T)[i]] )) / pi * 180))
    #     sd.angular.sep < - c(sd.angular.sep, sd  (acos(apply(cos.angle, 1, function(x) x[order(x, decreasing = T)[i]] )) / pi * 180) )
    # }
    # for i in range(2,127):
    #     np.append(mean_angular_sep, )
    cos_square = cos_angle ** 2

    # all-point smoother (256 points): include the observed point itself
    index = np.zeros((bvecs_64_expand.shape[0], bvecs_64.shape[0]), dtype=int)
    nearest_angle = weight_vonmises = np.zeros((bvecs_64_expand.shape[0], bvecs_64.shape[0]))
    scale = inv_scale = np.zeros(bvecs_64_expand.shape[0])
    for i in range(bvecs_64_expand.shape[0]):# i: index for gradient direction
        # index of 8 nearest neighborhood
        index[i, :] = np.argsort(-cos_angle[i, :])
        # 8 nearest angle
        nearest_angle[i, :] = arccos(cos_angle[i, index[i, :]])

        weight_vonmises[i, :] = vonmises.pdf(nearest_angle[i, :], kappa=5)
        scale[i] = np.sum(weight_vonmises[i, :])
        inv_scale[i] = 1 / scale[i]



    # assume values for the parameters, which can be obtained by manipulating different bval's.
    # s0 = est_s0 = 400
    # # sum.f=est.sum.f=0.5
    # b = bvals = 1500
    # d = 1 / 1500



