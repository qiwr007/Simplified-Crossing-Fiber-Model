import numpy as np
from numpy import cos, sin, pi, exp, sqrt, abs, round, arccos, arcsin
from scipy.stats import norm, vonmises, gamma
from scipy.optimize import fsolve
from scipy.special import expit
import matplotlib.pyplot as plt
from datetime import datetime
import random
import math
import os


# Transforms from cartesian coordinate to spherical coordinate
# according to paper, the elevation angle is theta, the azimuthal angle is phi
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


def rotation_matrix(theta, phi):
    # create rotation matrix
    ########## !!!!!!!!!!!! ###########
    # here has different with R code and paper, I take the R code version
    return np.array([[cos(theta) * cos(phi), -1 * sin(phi), sin(theta) * cos(phi)],
                    [cos(theta) * sin(phi), cos(phi), sin(theta) * sin(phi)],
                    [sin(theta), 0, -1 * cos(theta)]])


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
def SmootherEstimator_sim(vol_fracs, phis, kappa=None,  b=1500, d=1/1500, noise_level=0.05):
    # Nsim=1000; vol.fracs=c(0.3,0.3); phis=c(60, 120); kappa=10
    # Nsim: number of iterations
    # vol.fracs (vector): volume fraction for 2 fibers
    # phis (vector): phi angle for 2 fibers

    # 64-point observed intensity data: non-noisy, noisy
    # vol.fracs=c(0.2,0.7); phis=c(30,120); kappa=5
    # raw signal data across the true gradients (e.g. 64 directions)
    seed = np.random.choice(1000000, 1)[0]
    np.random.seed(seed)
    print("seed:", seed)
    file = open("seed.txt", "w+")
    file.write(str(seed))
    file.close()

    signal_data = BehrensSim(S0=400, f=vol_fracs, bval=np.full((bvecs_64.shape[0]), b), bvecs=bvecs_64, d=d, theta=np.array([0, 0]), phi=phis)

    # noise = np.random.normal(0, S0 * noise_level, int(bvecs_64.shape[0] / 2))
    noise_prime = norm.rvs(scale=S0 * noise_level, size=int(bvecs_64.shape[0] / 2))
    noise = np.append(noise_prime, noise_prime)
    # noise = norm.rvs(scale=S0 * noise_level, size=bvecs_64.shape[0])
    ################
    # !!!!!!!! output noise and draw the picture and seed
    signal_noise_data = signal_data + noise

    y_value = signal_noise_data

    ########################################
    #      estimator for sum.f and d       #
    ########################################
    # all-point smoother: smoothed signal data across all the expanded gradients (e.g. 128 directions),
    # taking linear combination of weighted observed data
    # if kappa has been assigned a value (e.g. not equal to 5 ),
    # the density function of Von-Mises should be changed
    ########################
    # Weiran think this part repeat with the main method, could be removed if we only use one kappa value
    if kappa is None or kappa > 999:  # revised "== 999" to "> 999"
        smooth_data1 = signal_noise_data
    else:
        for i in range(scale.shape[0]):
            weight_vonmises[i, :] = vonmises.pdf(nearest_angle[i, :], kappa=kappa)
            scale[i] = np.sum(weight_vonmises[i, :])
            inv_scale[i] = 1 / scale[i]
        biggg_vector = np.zeros(index.shape[0] * index.shape[1])
        dummy_index = 0
        for j in range(index.shape[1]):
            for i in range(index.shape[0]):
                biggg_vector[dummy_index] = signal_noise_data[index[i][j]]
                dummy_index += 1
        # index=nearest.angle=weight.vonmises= matrix(0,nrow=nrow(bvecs_64.expand), ncol=nrow(bvecs_64))
        # skip the transpose
        biggg_mat = biggg_vector.reshape((bvecs_64.shape[0], bvecs_64_expand.shape[0]))
        smooth_data1 = inv_scale * np.diag(weight_vonmises @ biggg_mat)

    ############################################################
    # some true value (We set the ground truth when simulating)
    true_sum_f = sum(vol_fracs)  # 0.3 + 0.5 = 0.8
    max_signal_data1 = S0 * ((1 - true_sum_f) * exp(-b * d) + true_sum_f)
    if kappa == None:
        max_signal_bias = 0
    else:
        max_signal_bias = max(smooth_data1) - max_signal_data1
    true_mean_signal = np.mean(signal_data)

    def equations(p):
        # list the set of the two equations, Eq(3) and Eq(4) in paper
        sum_f, b_mul_d = p
        return (1 - sum_f) * exp(-b_mul_d) + sum_f * sqrt(pi) * erf(sqrt(b_mul_d)) / (2 * sqrt(b_mul_d)) - np.mean(signal_noise_data) / S0, \
                (1 - sum_f) * exp(-b_mul_d) + sum_f - max(smooth_data1) / S0
    sum_f_solution, b_mul_d_solution = fsolve(equations, (0, 0.1))  # start point should be 0~1, otherwise may encounter computing error

    if kappa == None:  # if we do NOT use kernel smoothing
        est_sum_f = true_sum_f
        est_d = d
    else:  # if we use kernel smoothing
        est_sum_f = sum_f_solution
        est_d = b_mul_d_solution / b

    ########################################
    #      estimator for longitudinal      #
    ########################################
    # Here, the kappa is kappa_2 in paper, kappa_2 = 0.1
    for i in range(scale.shape[0]):
        weight_vonmises[i, :] = vonmises.pdf(nearest_angle[i, :], kappa=0.1)
        scale[i] = np.sum(weight_vonmises[i, :])
        inv_scale[i] = 1 / scale[i]
    # weight_vonmises = vonmises.pdf(nearest_angle, kappa=kappa)
    # scale = np.sum(weight_vonmises)
    # inv_scale = 1 / scale

    biggg_vector = np.zeros(index.shape[0] * index.shape[1])
    dummy_index = 0
    for j in range(index.shape[1]):
        for i in range(index.shape[0]):
            biggg_vector[dummy_index] = signal_noise_data[index[i][j]]
            dummy_index += 1
    # index=nearest.angle=weight.vonmises= matrix(0,nrow=nrow(bvecs_64.expand), ncol=nrow(bvecs_64))
    # skip the transpose
    biggg_mat = biggg_vector.reshape((bvecs_64.shape[0], bvecs_64_expand.shape[0]))
    smooth_data = inv_scale * np.diag(weight_vonmises @ biggg_mat)

    # bvecs_64_expand[smooth_data.argmax()] is the gradient direction of the strongest signal(after smoothing)
    # obtain the observed longitudinal axis
    obs_longitudinal_axis = bvecs_64_expand[smooth_data.argmax()]  # Equation (7) in paper
    obs_longitudinal_theta = cart2sph(obs_longitudinal_axis)[1]
    obs_longitudinal_phi = cart2sph(obs_longitudinal_axis)[0]
    if obs_longitudinal_theta < 0:
        obs_longitudinal_theta = -obs_longitudinal_theta
        obs_longitudinal_phi += pi  # here may exceed the arccos boundary


    # dot_product_directions = np.array([0, 0, 1]) @ bvecs_64_expand[smooth_data.argmax()]  # here has a difference with R code
    # if np.abs(dot_product_directions) > 1:
    #     dot_product_directions = 1
    # smooth_angle_biases = arccos(dot_product_directions) / pi * 180
    # if smooth_angle_biases > 90:
    #     smooth_angle_biases = 180 - smooth_angle_biases
    # dot.product.directions < - c(0, 0, 1) % * % bvecs_64.expand[which.max(smooth.data),]
    # if (abs(dot.product.directions) > 1) dot.product.directions < - round(dot.product.directions)  # > 1 is not possible
    # smooth.angle.bias < - acos(dot.product.directions) / pi * 180
    # if (smooth.angle.bias > 90) smooth.angle.bias=180-smooth.angle.bias
    # # smooth.angle.biases<- c(smooth.angle.biases, smooth.angle.bias)

    # rotate all gradient directions, as we rotate the observed longitudinal axis to Z-axis
    # update bvecs and r_x and r_y and length_r_xy and phi_r
    bvecs_64_prime = (AlignZ_Rot_mat(np.array([obs_longitudinal_theta, obs_longitudinal_phi])) @ bvecs_64.T).T
    r_x = bvecs_64_prime[:, 0]
    r_y = bvecs_64_prime[:, 1]
    length_r_xy = r_x ** 2 + r_y ** 2
    phi_r = np.arctan2(r_y, r_x)

    return TwoFiber_Behrens_simplified_non_informative_simulation_sim(y_value=y_value, b=b, est_sum_f=est_sum_f, est_d=est_d)
    # not end...


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


def TwoFiber_Behrens_simplified_non_informative_simulation_sim(y_value, b, est_sum_f, est_d):
    # some constant numbers
    NUM_SCAN = 100000  # number of MCMC scans
    THIN = 10  # thin interval

    # every THIN times iterations,,,
    # use est_1 to record the 4 estimated params (f1, phi1, phi2, sigma_sq)
    est_1 = np.empty((0, 4))
    # use log_s to record the 3 standard deviations for 3 params (f1, phi1, phi2)
    log_s = np.empty((0, 3))

    log_post_likelihood = np.empty((0, 1))

    # prior specification for sigma^2: Gibbs sampling in InverseGamma fucntion parameterized by shape.sigma, scale.sigma
    # set mean of error E(error)=0.005=1/theta*(k-1); k:shape, theta: scale
    # prior parameters for sigma^2
    shape_sigma = 101
    scale_sigma = 1400

    # starting values
    new_1 = np.array([0.0, pi, pi])  # this is the initial value of f1, phi_1, phi_2
    # it also used as Omege (in paper) (record the last adopted value) in the following iterations
    prop_1 = np.array([0.0, pi, pi])
    sigma_sq_1 = 14  # intial value for noise (sigma.square)

    # initialize log standard deviations for the component-wise proposal distributions
    # log(sd)
    log_s1 = np.full(3, -1.0)  # 3: number of params (except sigma.sq)

    # initialize acceptance rates
    ar_1 = np.zeros(3)

    # initialize acceptance counts
    accept_count_1 = np.zeros(3, dtype=int)

    counts = np.empty((0, 3))

    # here to record the system time
    start_time = datetime.now()
    # print("start at: ", start_time)

    for i in range(NUM_SCAN):
        if i % 1000 == 0:
            print(i)
        # get the batch number
        batch_num = i // 50 + 1  # batch size is 50
        # get the iteration number inside a batch
        iter_batch = i % 50
        delta = min(0.01, 1/sqrt(batch_num))

        # first iteration in a batch of 50,
        # so update acceptance rates, reset counters and update proposal variances
        if iter_batch == 0:
            # update the acceptance rates
            ar_1 = accept_count_1 / 50
            counts = np.append(counts, np.array([ar_1]), axis=0)

            # reset the acceptance counters
            accept_count_1 = np.zeros(3, dtype=int)

            # update variances of proposal function (which is a normal distribution, so that is has variances)
            # log_s1 is the log of variance, we will use exp() to get the variance
            for ii in range(len(log_s1)):
                if ar_1[ii] < 0.44:
                    log_s1[ii] = log_s1[ii] - delta
                else:
                    log_s1[ii] = log_s1[ii] + delta

        # update f1, phi1, phi2,   and count the accepted values
        for j in range(3):  # j is the component being updated
            # propose value from norm distribution
            # !!!!!! why ????????? is this because Bayesian posterior of uniform distribution??? #####
            # it may even generate a negative value !!!
            prop_1[j] = norm.rvs(size=1, loc=new_1[j], scale=exp(log_s1[j]))

            log_posterior_proposed = log_posterior_2(np.append(prop_1, sigma_sq_1), y_value, b, est_sum_f, est_d)
            log_posterior_Omega = log_posterior_2(np.append(new_1, sigma_sq_1), y_value, b, est_sum_f, est_d)
            log_mhratio_1 = log_posterior_proposed - log_posterior_Omega
            select_prob_1 = min(1, np.exp(log_mhratio_1))  # Here should not be exp, Weiran guess
            # according to MCMC.pdf P4, log_posterior_proposed = log[ P(x')Q(x|x') ]
            # log_posterior_Omega = log[ P(x)Q(x'|x) ], is this right??
            if random.random() < select_prob_1: #accept the proposal and increment the acceptance counter
                if j == 1:
                    new_1[j] = prop_1[j]
                else:
                    new_1[j] = prop_1[j] % pi  # ensure it in range [0,pi)
                accept_count_1[j] += 1

        # update sigma.sq (standard deviation of noise) using Gibbs
        # f1 updated by logistic sigmoid function: expit(x) = 1/(1+exp(-x)
        # but why f1 can be updated like this?
        # f1 = est_sum_f / (1 + exp(-new_1[0]))
        f1 = expit(new_1[0]) * est_sum_f
        if i % 1000 == 0:
            print(f1)
            print(new_1[0])
        f2 = est_sum_f - f1
        phi1 = new_1[1]
        phi2 = new_1[2]

        temp_1 = f1 * exp(-b * est_d * length_r_xy * (cos(phi_r - phi1) ** 2)) + f2 * exp(-b * est_d * length_r_xy * (cos(phi_r-phi2) ** 2))  # stick component
        fit_y_value = est_S0 * ((1 - est_sum_f) * exp(-b * est_d) + temp_1)
        temp_2 = (y_value - fit_y_value) ** 2
        temp_3 = sum(temp_2)

        # !!!!!!!!!!!????????? ############
        # update sigma_square here, I am so confused about why we update like this??
        new_scale_sigma = 1 / (temp_3 / 2 + 1 / scale_sigma)
        new_shape_sigma = shape_sigma + y_value.shape[0]
        # according to Wiki, the mean of Gamma distribution is shape * scale
        # many questions here !!!!!!! ################
        sigma_sq_1 = 1 / gamma.rvs(1, scale=new_scale_sigma, loc=new_shape_sigma * new_scale_sigma)

        if i % THIN == 0:
            est_1 = np.append(est_1, np.array([np.append(new_1, sigma_sq_1)]), axis=0)
            log_s = np.append(log_s, np.array([log_s1]), axis=0)
            log_post_likelihood = np.append(log_post_likelihood, log_posterior_Omega)  # log_post_likelihood vector

    # End of iterations
    end_time = datetime.now()
    run_time = end_time - start_time
    print(run_time)  # show the running time
    file = open("run_time.txt", "w+")
    file.write(str(run_time))
    file.close()

    NUM_EST = est_1.shape[0]  # how many estimates we have
    # NUM_BURN should be related to accept_count, "counts"
    BURN = math.floor(NUM_EST / 2)  # how much of them to discard as burn-in

    plt.plot(est_1[:, 0])
    sample_file_name = "est_f1"
    plt.savefig(results_dir + sample_file_name)
    plt.close()

    plt.plot(est_1[:, 1])
    sample_file_name = "est_phi1"
    plt.savefig(results_dir + sample_file_name)
    plt.close()

    plt.plot(est_1[:, 2])
    sample_file_name = "est_phi2"
    plt.savefig(results_dir + sample_file_name)
    plt.close()

    plt.plot(est_1[:, 3])
    sample_file_name = "est_sigma_sq"
    plt.savefig(results_dir + sample_file_name)
    plt.close()

    return est_1


# function used in function "TwoFiber_Behrens_simplified_non_informative_simulation_sim",
# to calculate the posterior log-likelihood in MCMC sampling
# log_posterior_2(np.append(prop_1, sigma_sq_1), y_value, b, est_sum_f, est_d)
# it uses global variance: length_r_xy, phi_r,
def log_posterior_2(params, y_value, b, est_sum_f, est_d):
    # (1)
    # f1 = est_sum_f * 1 / (1 + exp(-1 * params[0]))
    f1 = est_sum_f * expit(params[0])
    # (2)
    f2 = est_sum_f - f1
    phi1 = params[1]
    phi2 = params[2]
    sigma_sq = params[3]
    temp_1 = f1 * exp(-b * est_d * length_r_xy * ((cos(phi_r - phi1)) **2)) + f2 * exp(-b * est_d * length_r_xy * (cos(phi_r - phi2) ** 2))
    fit_y_value = est_S0 * ((1-est_sum_f) * exp(-b * est_d) + temp_1)
    temp_2 = -1 * (y_value - fit_y_value) ** 2 / (2 * sigma_sq)
    temp_3 = sum(temp_2)
    # non-informative_simulation prior specification
    return temp_3

if __name__ == '__main__':
    # the output PATH
    script_dir = os.path.dirname(__file__)  # this path
    results_dir = os.path.join(script_dir, 'Results/')  # the name of folder

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # bvals = read.readFile('dataset/bvals')
    # bvecs = read.readFile('dataset/bvecs')

    bvecs_jones = np.loadtxt('dataset/bvecs_jones.txt')
    bvecs_jones = bvecs_jones[7:71]
    # becuase we have 64 bvecs gradients from Jones, set BVECS = 64
    BVECS = 64
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
    bvecs_64 = (AlignZ_Rot_mat(np.array([middle_direction_theta, middle_direction_phi])) @ bvecs_64_prime.T).T
    for i in range(bvecs_64.shape[0]):
        bvecs_64[i] = bvecs_64[i] / np.sqrt(np.sum(bvecs_64[0]**2))

    # create the hypothetical gradients amid the true gradients
    reference_dir = bvecs_64[np.argmax(bvecs_64[:, 2])]
    reference_dir_sph = cart2sph(reference_dir)
    reference_dir_theta = reference_dir_sph[1]
    reference_dir_phi = reference_dir_sph[0]
    bvecs_64_hypo = (AlignZ_Rot_mat(np.array([reference_dir_theta, reference_dir_phi])) @ bvecs_64.transpose()).T
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
            cos_angle[i][j] = bvecs_64_expand[i, :] @ bvecs_64[j, :]
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
    nearest_angle = np.zeros((bvecs_64_expand.shape[0], bvecs_64.shape[0]))
    weight_vonmises = np.zeros((bvecs_64_expand.shape[0], bvecs_64.shape[0]))
    scale = np.zeros(bvecs_64_expand.shape[0])
    inv_scale = np.zeros(bvecs_64_expand.shape[0])
    for i in range(bvecs_64_expand.shape[0]):# i: index for gradient direction
        # index is: the index of element in the i-th row of "cos_angle", from the biggest to the smallest
        index[i] = np.argsort(-cos_angle[i])
        # 8 nearest angle
        nearest_angle[i] = arccos(cos_angle[i, index[i]])

        weight_vonmises[i] = vonmises.pdf(nearest_angle[i], kappa=5)
        scale[i] = np.sum(weight_vonmises[i])
        inv_scale[i] = 1 / scale[i]

    # assume values for the parameters, which can be obtained by manipulating different bval's.
    S0 = est_S0 = 400
    # sum.f=est.sum.f=0.5
    b = bvals = 1500
    d = 1 / 1500
    vol_fracs = np.array([0.3, 0.5])  # f1 = 0.3, f2 = 0.5
    kappa_value = 20
    phi1 = 45
    phi2 = 180 - phi1
    phis = np.array([phi1, phi2])

    output = SmootherEstimator_sim(vol_fracs=vol_fracs, kappa=kappa_value, phis=phis, noise_level=0.05)
