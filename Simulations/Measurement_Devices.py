import Line
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from Line import Line, Line_2D
import math as m
import sys
import numpy as np
import nifty8 as ift
import plotly.graph_objects as go
import math as m
from scipy.optimize import minimize, least_squares
import scipy.ndimage
from time import time




class Measurement_Devices:
    """
    contains class methods for the DOAS devices:
    -> measure
    -> Add Doas
    """

    def __init__(self, DOAS_devices, Reflectors, simulated_field):
        self.DOAS_devices = DOAS_devices
        self.Reflectors = Reflectors
        self.simulated_field = simulated_field
        self.normalized_field = np.sum(self.simulated_field, axis=2) / np.max(np.sum(self.simulated_field, axis=2))

    def print_parameters(self):
        for DOAS_device in self.DOAS_devices:
            DOAS_device.print_parameters()
        for Reflector in self.Reflectors:
            Reflector.print_parameters()
        #print("simulated_field:", self.simulated_field)

    def return_plottables(self):
        return_arr = []
        for DOAS_device in self.DOAS_devices:
            for Reflector in self.Reflectors:
                line_points = Line_2D(Reflector.position-DOAS_device.position, DOAS_device.position).get_plot_points()
                #print(line_points)
                return_arr.append(line_points)
        return return_arr

    def return_positions(self, dim="3D"):
        doas_positions = []
        for DOAS_device in self.DOAS_devices:
            doas_positions.append(DOAS_device.position)
        refl_positions = []
        for Reflector in self.Reflectors:
            refl_positions.append(Reflector.position)
        if dim=="3D":
            return doas_positions, refl_positions
        else:
            return [k[:2] for k in doas_positions], [k[:2] for k in refl_positions]

    def measure_3D(self):
        """
        measure in the simulated field along a straight line
        :return: b-vector of the measurement
        """
        # 1. orientation and position -> line-equation
        # 2. select voxels from simulated field
        # 3. add the value together and add noise
        # 4. store the values in a b-vector and display as pandas-dataframe together with information about
        #       ID, Position, Orientation, Measured Value, Uncertainty
        #

        # for DOAS_device in DOAS_devices:
        #    orientation = DOAS_device.orientation
        #    position = DOAS_device.position
        #    ID = DOAS_device.ID
        #    x_len, y_len, z_len = simulated_field.shape
        #    t = np.linspace(0,x_len,100)
        #    print(line_eq(t, position, orientation))
        measurements=[]
        #fig = plt.figure()
        #ax = fig.gca(projection='3d')
        #ax.set_aspect('auto')
        lines = self.return_plottables()
        print(lines)

        doas_ids = [DOAS_device.ID for DOAS_device in self.DOAS_devices]
        reflector_ids = [refl.ID for refl in self.Reflectors]
        #print(doas_ids, reflector_ids)
        doas_idx = 0
        reflector_idx = 0
        for line in lines:
            coordinates=[]
            for i in range(line.shape[1]):
                #print(line[:,i].astype("int"))
                coordinates.append(tuple(line[:,i].astype("int").tolist()))
            coordinates = list(dict.fromkeys(coordinates))
            print(coordinates)
            x_len=self.simulated_field.shape[0]
            y_len=self.simulated_field.shape[1]
            z_len=self.simulated_field.shape[2]
            X, Y, Z = np.mgrid[:x_len, :y_len, :z_len]
            vol = np.zeros((x_len, y_len, z_len))
            for x, y, z in coordinates:
                if x>0 and y>0 and z>0:
                    vol[x, y, z] = 1
            #plot_data = [go.Volume(
            #    x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            #    value=vol.flatten(),
            #    isomin=0.9,
            #    isomax=1.,
            #    opacity=1,
            #    surface_count=1,
            #)]
            #fig = go.Figure(data=plot_data)#

            #fig.update_layout(scene_xaxis_showticklabels=False,
            #                  scene_yaxis_showticklabels=False,
            #                  scene_zaxis_showticklabels=False)
            #fig.show()
            weights = np.multiply(self.simulated_field, vol)
            #print(weights[np.nonzero(weights)])
            #ax.voxels(weights, edgecolor="k")
            print("Doas {} to Reflector {}: {} ppb".format(doas_ids[doas_idx], reflector_ids[reflector_idx],
                                                          round(np.sum(weights[np.nonzero(weights)]), 5)))
            measurements.append(np.sum(weights[np.nonzero(weights)]))

            reflector_idx += 1
            if reflector_idx == len(reflector_ids):
                reflector_idx=0
                doas_idx+=1


        #plt.show()
        self.measurement = measurements
        self.measured_lines = lines

        return measurements

    def measure_2D(self):
        """
        measure in the simulated field along a straight line
        :return: b-vector of the measurement
        """

        measurements=[]

        lines = self.return_plottables()

        doas_ids = [DOAS_device.ID for DOAS_device in self.DOAS_devices]
        reflector_ids = [refl.ID for refl in self.Reflectors]
        doas_idx = 0
        reflector_idx = 0
        weights_total = np.zeros(self.normalized_field.shape)
        for line in lines:
            coordinates = []
            for i in range(line.shape[1]):
                coordinates.append(tuple(line[:,i].astype("int").tolist()))
            coordinates = list(dict.fromkeys(coordinates))

            x_len = self.normalized_field.shape[0]
            y_len = self.normalized_field.shape[1]
            vol = np.zeros((x_len, y_len))
            for x, y in coordinates:
                if x > 0 and y > 0:
                    vol[x, y] = 1

            weights = np.multiply(self.normalized_field, vol)
            #weights_total[np.where(weights > 0)] = 1
            weights_total += vol * np.sum(weights)
            #print(weights[np.nonzero(weights)])

            print("Doas {} to Reflector {}: {} ppb".format(doas_ids[doas_idx], reflector_ids[reflector_idx],
                                                           round(np.sum(weights[np.nonzero(weights)]), 5)))
            measurements.append(np.sum(weights[np.nonzero(weights)]))

            reflector_idx += 1
            if reflector_idx == len(reflector_ids):
                reflector_idx=0
                doas_idx+=1
        print("-----------------------------------------------------------------------")
        self.measurement = measurements
        self.measured_lines = lines
        self.measurement_weights_plotting = weights_total

        return measurements

    def gaussian_inversion_2D(self, show=False):

        def measure_2D_num(field):
            """
            measure in the simulated field along a straight line
            :return: b-vector of the measurement
            """
            measurements = []

            def return_plottables_num():
                return_arr = []
                for DOAS_device in self.DOAS_devices:
                    for Reflector in self.Reflectors:
                        line_points = Line_2D(Reflector.position - DOAS_device.position, DOAS_device.position).get_plot_points()
                        return_arr.append(line_points)
                return return_arr
            lines = return_plottables_num()

            doas_ids = [DOAS_device.ID for DOAS_device in self.DOAS_devices]
            reflector_ids = [refl.ID for refl in self.Reflectors]
            doas_idx = 0
            reflector_idx = 0
            weights_total = np.zeros(field.shape)
            for line in lines:
                coordinates = []
                for i in range(line.shape[1]):
                    coordinates.append(tuple(line[:, i].astype("int").tolist()))
                coordinates = list(dict.fromkeys(coordinates))

                x_len = field.shape[0]
                y_len = field.shape[1]
                vol = np.zeros((x_len, y_len))
                for x, y in coordinates:
                    if x > 0 and y > 0:
                        vol[x, y] = 1

                weights = np.multiply(field, vol)
                weights_total += vol * np.sum(weights)
                measurements.append(np.sum(weights[np.nonzero(weights)]))

                reflector_idx += 1
                if reflector_idx == len(reflector_ids):
                    reflector_idx = 0
                    doas_idx += 1

            return measurements, weights_total

        def gaussian_rotatable(mux, muy, sigmax, sigmay, theta):
            grid_x, grid_y = np.mgrid[0:29:30j, 0:29:30j]
            theta = np.deg2rad(theta)
            val = np.exp(-0.5 * ((((grid_x - mux) * np.cos(theta) -
                 (grid_y - muy) * np.sin(theta)) / sigmax) ** 2 + (((grid_x - mux) * np.sin(theta) +
                 (grid_y - muy) * np.cos(theta)) / sigmay) ** 2)) # * 1/(sigmax * 2 * np.pi * sigmay)
            return val



        def measure_2D_NumInt(field, doas, refl, mode, show=False, field_params=None):
            x1,x0,y1,y0 = [refl[0], doas[0], refl[1], doas[1]]
            length = int(np.hypot(x1 - x0, y1 - y0))*100
            x, y = np.linspace(x0, x1, length), np.linspace(y0, y1, length)
            if mode == "Testing":
                mux, muy, sigmax, sigmay = field_params
                ana_val, ana_arr = integr_gaussian(mux, muy, sigmax, sigmay, doas, refl, full=True)
                plt.style.use(['science', 'no-latex'])
                zi = scipy.ndimage.map_coordinates(field, np.vstack((x, y)))
                fig, axes = plt.subplots(nrows=2, figsize=(5, 10))

                axes[0].imshow(field.T, cmap="inferno")
                axes[0].plot([x0, x1], [y0, y1], 'wo-')
                axes[0].axis('image')
                axes[1].plot(zi, label="Cubic Interp.")

                zi_NN = field[x.astype(np.int), y.astype(np.int)]
                plt.title(f"Cub: {round(np.sum(zi), 3)}, NN: {round(np.sum(zi_NN), 3)}, Ana: {round(np.sum(ana_val), 3)}")
                axes[1].plot(zi_NN, label="Nearest Neighb. Interp.")
                plt.grid(alpha=0.3)
                plt.legend()
                #plt.show()
                print(f"Cub: {round(np.sum(zi), 3)}, NN: {round(np.sum(zi_NN), 3)}, Ana: {round(np.sum(ana_val), 3)}")
                return np.sum(zi_NN)

            elif mode == "cubic":
                zi = scipy.ndimage.map_coordinates(field, np.vstack((x,y)))
                #mux, muy, sigmax, sigmay = field_params
                #ana_val, ana_arr = integr_gaussian(mux, muy, sigmax, sigmay, doas, refl, full=True)
                #print(f"[{round(ana_val, 3)}, {round(np.sum(zi)/100, 3)}],")
                if show:
                    fig, axes = plt.subplots(nrows=2)
                    axes[0].imshow(field.T, cmap="inferno")
                    axes[0].plot([x0, x1], [y0, y1], 'wo-')
                    axes[0].axis('image')
                    axes[1].plot(zi)
                    plt.show()
                return np.sum(zi)/100

            elif mode == "direct":
                zi = field[x.astype(np.int), y.astype(np.int)]
                if show:
                    fig, axes = plt.subplots(nrows=2)
                    axes[0].imshow(field.T, cmap="inferno")
                    axes[0].plot([x0, x1], [y0, y1], 'wo-')
                    axes[0].axis('image')
                    axes[1].plot(zi)
                    plt.show()
                return np.sum(zi)

        def integr_gaussian(mux, muy, sigmax, sigmay, doas, refl, full=False):
            """
            Integrated Multivariate Gaussian function.
            :param mux: mu in x-direction
            :param muy: mu in y-direction
            :param sigmax: sigma in x-direction
            :param sigmay: sigma in y-direction
            :param doas: 2-D array of DOAS position
            :param refl: 2-D array of REFLECTOR position
            :return: returns integrated value along a line-vector
            """

            def custom_erf(t):
                return m.erf((rx*sigmay**2*(kx+rx*t) + ry*sigmax**2*(ky+ry*t)) /
                             (np.sqrt(2)*sigmax*sigmay*np.sqrt(rx**2*sigmay**2 + ry**2*sigmax**2)))

            rx, ry = [refl[0]-doas[0], refl[1]-doas[1]]
            kx, ky = [doas[0]-mux, doas[1]-muy]
            C = np.sqrt(rx**2+ry**2) /(2*np.sqrt(rx**2*sigmay**2 + ry**2*sigmax**2)) * \
                np.exp(-(ky*rx - kx*ry)**2 / (2*(rx**2*sigmay**2 + ry**2*sigmax**2)))
            if not full:
                return (custom_erf(1) - custom_erf(0)) * C * (np.sqrt(2*np.pi)*sigmay*sigmax)
            else:
                return (custom_erf(1) - custom_erf(0)) * C * (np.sqrt(2 * np.pi) * sigmay * sigmax), [custom_erf(t) * C * (np.sqrt(2 * np.pi) * sigmay * sigmax) for t in np.linspace(0,29,30)]

        def get_numerical_measurements(mux, muy, sigmax, sigmay, lines, doas, refl):
            weights_total = np.zeros(self.normalized_field.shape)
            j = 0
            for doas in doas:
                for i in range(len(refl)):
                    coordinates = []
                    for k in range(lines[j].shape[1]):
                        coordinates.append(tuple(lines[j][:, k].astype("int").tolist()))
                    coordinates = list(dict.fromkeys(coordinates))
                    x_len = self.normalized_field.shape[0]
                    y_len = self.normalized_field.shape[1]
                    vol = np.zeros((x_len, y_len))
                    for x, y in coordinates:
                        if x > 0 and y > 0:
                            vol[x, y] = 1
                    j += 1
                    print(f"Doas {doas.ID} to Reflector {refl[i].ID}: {round(integr_gaussian(mux, muy, sigmax, sigmay, doas.position, refl[i].position), 5)} ppb")
                    weights = vol * integr_gaussian(mux, muy, sigmax, sigmay, doas.position, refl[i].position)
                    weights_total += weights
            return weights_total

        def gaussian(mux, muy, sigmax, sigmay, A = 1):
            grid_x, grid_y = np.mgrid[0:29:30j, 0:29:30j]
            return A * np.exp(-(((grid_x - mux) ** 2 / (2 * sigmax ** 2)) + ((grid_y - muy) ** 2 / (2 * sigmay ** 2))))

        def get_number_of_gaussians(number_of_doas, number_of_refl):
            #print(f"# Doas {number_of_doas}, # Refl {number_of_refl} -> {int((number_of_doas * number_of_refl)/5)} gaussians")
            return int((number_of_doas * number_of_refl)/5)

        def get_dict_of_gaussian_params(number_of_gaussians):
            gaussian_params = {}
            for i in range(number_of_gaussians):
                mus_x = "mux%d" % i
                mus_y = "muy%d" % i
                sigmas_x = "sigmax%d" % i
                sigmas_y = "sigmay%d" % i
                thetas = "theta%d" % i
                gaussian_params[mus_x] = np.random.randint(1, 28)
                gaussian_params[mus_y] = np.random.randint(1, 28)
                gaussian_params[sigmas_x] = np.random.randint(2, 10)
                gaussian_params[sigmas_y] = np.random.randint(2, 10)
                gaussian_params[thetas] = np.random.randint(0, 359)
            print(f"Fittable Gaussian Parameters: {gaussian_params}")
            return gaussian_params

        def create_param_bounds(number_of_gaussians, mu_min, mu_max, sigma_min, sigma_max, theta_min, theta_max):
            return_list = []
            for i in range(number_of_gaussians):
                return_list.append((mu_min, mu_max))
                return_list.append((mu_min, mu_max))
                return_list.append((sigma_min, sigma_max))
                return_list.append((sigma_min, sigma_max))
                return_list.append((theta_min, theta_max))
            print(return_list)
            return return_list

        def diff(params, *args):
            if len(params) == 4:
                mux, muy, sigmax, sigmay = params
            try:
                doas, refl, measurements, proove_arr, numerical = args
            except:
                doas, refl, measurements, proove_arr, numerical = args[0]
            if not numerical:
                diff_arr = []
                y=0
                for doas in doas:
                    for i in range(len(refl)):
                        sim_meas = integr_gaussian(mux, muy, sigmax, sigmay, doas.position, refl[i].position)
                        diff_arr.append(abs(measurements[y] - sim_meas) ** 5)
                        y+=1
                cost_func_vals_ana.append(np.sum(diff_arr))
                return np.sum(diff_arr)
            else:
                diff_arr = []
                y = 0
                for doas in doas:
                    for i in range(len(refl)):
                        field = np.zeros(self.normalized_field.shape)
                        for k in range(get_number_of_gaussians(len(self.DOAS_devices), len(self.Reflectors))):
                            field += gaussian_rotatable(params[k*5+0], params[k*5+1], params[k*5+2], params[k*5+3], params[k*5+4])
                        sim_meas = measure_2D_NumInt(field, doas.position, refl[i].position, "cubic", show=False, field_params=None)
                        diff_arr.append(abs(measurements[y] - sim_meas) ** 5)
                        y += 1
                cost_func_vals_num.append(np.sum(diff_arr))
                print(np.sum(diff_arr))
                return np.sum(diff_arr)

        if show:
            number_of_gaussians = get_number_of_gaussians(len(self.DOAS_devices), len(self.Reflectors))
            gaussian_params = get_dict_of_gaussian_params(number_of_gaussians)

            cost_func_vals_ana = []
            start_ana = time()
            res_analytical = minimize(diff, np.array([10, 20, 3, 6]),
                                      args=[self.DOAS_devices, self.Reflectors, self.measurement, cost_func_vals_ana, False],
                                      bounds=[(1,28), (1,28), (1,10), (1,10)])
            end_ana = time()
            time_ana = end_ana-start_ana
            print(res_analytical)
            mux_ana, muy_ana, sigmax_ana, sigmay_ana = res_analytical.x

            cost_func_vals_num = []
            start_num = time()

            res_numerical = minimize(diff, np.array(list(gaussian_params.values())),
                                     args=[self.DOAS_devices, self.Reflectors, self.measurement, cost_func_vals_num, True],
                                     bounds=create_param_bounds(number_of_gaussians, 1, 28, 2, 10, 0, 359))
            end_num = time()
            time_num = end_num - start_num
            print(res_numerical)
            num_resulting_field = np.zeros(self.normalized_field.shape)
            for k in range(get_number_of_gaussians(len(self.DOAS_devices), len(self.Reflectors))):
                num_resulting_field += gaussian_rotatable(res_numerical.x[k*4 + 0], res_numerical.x[k*4 + 1], res_numerical.x[k*4 + 2], res_numerical.x[k*4 + 3], res_numerical.x[k*4 + 4])


            plt.rcParams['text.usetex'] = True
            fig = plt.figure()

            fig.add_subplot(331)
            plt.imshow(self.normalized_field, cmap="inferno")
            plt.colorbar()
            plt.title("Ground Truth")

            fig.add_subplot(332)
            plt.imshow(self.measurement_weights_plotting, cmap="inferno")
            plt.colorbar()
            plt.title("Measurements Ground Truth")

            fig.add_subplot(333)
            plt.axis("off")

            fig.add_subplot(334)
            plt.imshow(gaussian(mux_ana, muy_ana, sigmax_ana, sigmay_ana), cmap="inferno")
            plt.colorbar()
            plt.title("Retrieved Field analytical")

            fig.add_subplot(335)
            plt.imshow(get_numerical_measurements(mux_ana, muy_ana, sigmax_ana, sigmay_ana, self.measured_lines, self.DOAS_devices, self.Reflectors), cmap="inferno")
            plt.colorbar()
            plt.title("Measurements Retr. Field analytical")

            fig.add_subplot(336)
            plt.plot(np.array(cost_func_vals_ana))
            plt.yscale("log")
            plt.xlabel("$\#$ fev")
            plt.ylabel("$\chi^2$")
            plt.grid(True, which="both", ls="-", alpha=0.3)
            plt.title(r"$\chi^2$ analytical " + str(round(time_ana,5)) + "s")

            fig.add_subplot(337)
            plt.imshow(num_resulting_field, cmap="inferno")
            plt.colorbar()
            plt.title("Retrieved Field numerical")

            fig.add_subplot(338)
            plt.imshow(measure_2D_num(num_resulting_field)[1], cmap="inferno")
            plt.colorbar()
            plt.title("Measurements Retr. Field numerical")

            fig.add_subplot(339)
            plt.plot(np.array(cost_func_vals_num))
            plt.yscale("log")
            plt.xlabel("$\#$ fev")
            plt.ylabel("$\chi^2$")
            plt.grid(True, which="both", ls="-", alpha=0.3)
            plt.title(r"$\chi^2$ numerical " + str(round(time_num,5)) + "s")

            plt.show()


    def IFT8_inversion_2D(self):
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            master = comm.Get_rank() == 0
        except ImportError:
            comm = None
            master = True

        def build_LOSs(doas_pos, refl_pos, normalization):
            LOS_starts = [[], []]
            LOS_ends = [[], []]
            for doas in doas_pos:
                for i in range(len(refl_pos)):
                    LOS_starts[0].append(doas[0] / normalization)
                    LOS_starts[1].append(doas[1] / normalization)
                    LOS_ends[0].append(refl_pos[i][0] / normalization)
                    LOS_ends[1].append(refl_pos[i][1] / normalization)
            return LOS_starts, LOS_ends

        filename = "./output_ift_inversion/testing_field_inversion_{}.png"
        position_space = ift.RGSpace((self.simulated_field.shape[0], self.simulated_field.shape[1]))
        normalized_field = (np.sum(self.simulated_field, axis=2) / np.max(np.sum(self.simulated_field, axis=2)))
        padder = ift.FieldZeroPadder(position_space, [i * 2 for i in position_space.shape], central=False).adjoint
        print(padder.domain)
        #  For a detailed showcase of the effects the parameters
        #  of the CorrelatedField model have on the generated fields,
        #  see 'getting_started_4_CorrelatedFields.ipynb'.

        args = {
            'offset_mean': 0,
            'offset_std': (1e-3, 1e-6),
            # Amplitude of field fluctuations
            'fluctuations': (1., 0.1),  # 1.0, 1e-2
            # Exponent of power law power spectrum component
            'loglogavgslope': (-4., 0.5),  # -6.0, 1   (mean, std)
            # Amplitude of integrated Wiener process power spectrum component
            'flexibility': (0.2, 0.1),  # 1.0, 0.5
            # How ragged the integrated Wiener process component is
            'asperity': (0.5, 0.4)  # 0.1, 0.5
        }

        correlated_field = ift.SimpleCorrelatedField(padder.domain, **args)
        pspec = correlated_field.power_spectrum
        print(correlated_field.domain)
        # ift.random.push_sseq_from_seed(np.random.randint(0,100))

        # Apply a nonlinearity
        # signal = ift.sigmoid(correlated_field) #das hier sollte ich ??ndern!
        signal = ift.log(ift.Adder(1., domain=position_space)(ift.exp(padder(correlated_field))))

        # Build the line-of-sight response and define signal response
        doas_pos = self.return_positions(dim="2D")[0]
        refl_pos = self.return_positions(dim="2D")[1]
        LOS_starts, LOS_ends = build_LOSs(doas_pos, refl_pos, 30)
        R = ift.LOSResponse(position_space, starts=LOS_starts, ends=LOS_ends)
        signal_response = R(signal)


        # Specify noise
        data_space = R.target
        noise = .1
        N = ift.ScalingOperator(data_space, noise, np.float64)

        # Generate mock signal and data
        # mock_position = ift.from_random(signal_response.domain, 'normal')
        ground_truth = ift.makeField(position_space, normalized_field)

        # plot = ift.Plot()
        # plot.add(R(ground_truth), title='Ground Truth', zmin=0, zmax=1)
        # plot.output()

        data = R(ground_truth) + N.draw_sample()

        # Minimization parameters
        ic_sampling = ift.AbsDeltaEnergyController(name="Sampling (linear)",
                                                   deltaE=0.005, iteration_limit=100)
        ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.05,
                                                 convergence_level=2, iteration_limit=35)
        minimizer = ift.NewtonCG(ic_newton)
        ic_sampling_nl = ift.AbsDeltaEnergyController(name='Sampling (nonlin)',
                                                      deltaE=0.05, iteration_limit=15, convergence_level=2)
        minimizer_sampling = ift.NewtonCG(ic_sampling_nl)

        # Set up likelihood energy and information Hamiltonian
        likelihood_energy = (ift.GaussianEnergy(data, inverse_covariance=N.inverse) @ signal_response)
        H = ift.StandardHamiltonian(likelihood_energy, ic_sampling)

        initial_mean = ift.MultiField.full(H.domain, 0.)
        mean = initial_mean

        plot = ift.Plot()
        plot.add(ground_truth, title='Ground Truth', zmin=0, zmax=1)
        plot.add(R.adjoint_times(data), title='Data')
        plot.add([pspec.force(mean)], title='Power Spectrum')
        plot.output(ny=1, nx=3, xsize=24, ysize=6, name=filename.format("setup"))

        # number of samples used to estimate the KL
        # Minimize KL
        n_iterations = 6
        n_samples = lambda iiter: 10 if iiter < 5 else 20
        samples = ift.optimize_kl(likelihood_energy, n_iterations, n_samples,
                                  minimizer, ic_sampling, minimizer_sampling,
                                  plottable_operators={"signal": (signal, dict(vmin=0, vmax=1)),
                                                       "power spectrum": pspec},
                                  ground_truth_position=None,
                                  output_directory="output_ift_inversion",
                                  overwrite=True, comm=comm)

        if True:
            # Load result from disk. May be useful for long inference runs, where
            # inference and posterior analysis are split into two steps
            samples = ift.ResidualSampleList.load("output_ift_inversion/pickle/last", comm=comm)

        # Plotting
        filename_res = filename.format("results")
        plot = ift.Plot()
        mean, var = samples.sample_stat(signal)
        plot.add(mean, title="Posterior Mean", vmin=0, vmax=1)
        plot.add(var.sqrt(), title="Posterior Standard Deviation", vmin=0)

        nsamples = samples.n_samples
        logspec = pspec.log()
        #plot.add(list(samples.iterator(pspec)) +
        #         [pspec.force(ground_truth), samples.average(logspec).exp()],
        #         title="Sampled Posterior Power Spectrum",
        #         linewidth=[1.] * nsamples + [3., 3.],
        #         label=[None] * nsamples + ['Ground truth', 'Posterior mean'])
        return mean.val.T, var.sqrt().val.T


    def IFT8_inversion_3D(self):
        try:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            master = comm.Get_rank() == 0
        except ImportError:
            comm = None
            master = True

        def build_LOSs(doas_pos, refl_pos, normalization):
            LOS_starts = [[], [], []]
            LOS_ends = [[], [], []]
            for doas in doas_pos:
                print(doas)
                for i in range(len(refl_pos)):
                    LOS_starts[0].append(doas[0] / normalization)
                    LOS_starts[1].append(doas[1] / normalization)
                    LOS_starts[2].append(doas[2] / normalization)
                    LOS_ends[0].append(refl_pos[i][0] / normalization)
                    LOS_ends[1].append(refl_pos[i][1] / normalization)
                    LOS_ends[2].append(refl_pos[i][2] / normalization)
            return LOS_starts, LOS_ends

        filename = "./output_ift_inversion/testing_field_inversion_{}.png"
        position_space = ift.RGSpace((self.simulated_field.shape))
        padder = ift.FieldZeroPadder(position_space, [i * 2 for i in position_space.shape], central=False).adjoint
        print(padder.domain)
        #  For a detailed showcase of the effects the parameters
        #  of the CorrelatedField model have on the generated fields,
        #  see 'getting_started_4_CorrelatedFields.ipynb'.

        args = {
            'offset_mean': 0,
            'offset_std': (1e-3, 1e-6),
            # Amplitude of field fluctuations
            'fluctuations': (1., 0.1),  # 1.0, 1e-2
            # Exponent of power law power spectrum component
            'loglogavgslope': (-3., 0.5),  # -6.0, 1   (mean, std)
            # Amplitude of integrated Wiener process power spectrum component
            'flexibility': (0.5, 0.3),  # 1.0, 0.5
            # How ragged the integrated Wiener process component is
            'asperity': (0.5, 0.4)  # 0.1, 0.5
        }

        correlated_field = ift.SimpleCorrelatedField(padder.domain, **args)
        pspec = correlated_field.power_spectrum
        print(correlated_field.domain)
        # ift.random.push_sseq_from_seed(np.random.randint(0,100))

        # Apply a nonlinearity
        # signal = ift.sigmoid(correlated_field) #das hier sollte ich ??ndern!
        signal = ift.log(ift.Adder(1., domain=position_space)(ift.exp(padder(correlated_field))))

        # Build the line-of-sight response and define signal response
        doas_pos = self.return_positions(dim="3D")[0]
        refl_pos = self.return_positions(dim="3D")[1]
        LOS_starts, LOS_ends = build_LOSs(doas_pos, refl_pos, 30)
        R = ift.LOSResponse(position_space, starts=LOS_starts, ends=LOS_ends)
        signal_response = R(signal)


        # Specify noise
        data_space = R.target
        noise = .001
        N = ift.ScalingOperator(data_space, noise, np.float64)

        # Generate mock signal and data
        # Ich m??chte normalized_field als mock field benutzen.
        # mock_position = ift.from_random(signal_response.domain, 'normal')
        ground_truth = ift.makeField(position_space, self.simulated_field)
        # ground_truth = signal(ift.from_random(signal.domain))

        # plot = ift.Plot()
        # plot.add(R(ground_truth), title='Ground Truth', zmin=0, zmax=1)
        # plot.output()

        data = R(ground_truth) + N.draw_sample()

        # Minimization parameters
        ic_sampling = ift.AbsDeltaEnergyController(name="Sampling (linear)",
                                                   deltaE=0.05, iteration_limit=100)
        ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.5,
                                                 convergence_level=2, iteration_limit=35)
        minimizer = ift.NewtonCG(ic_newton)
        ic_sampling_nl = ift.AbsDeltaEnergyController(name='Sampling (nonlin)',
                                                      deltaE=0.5, iteration_limit=15, convergence_level=2)
        minimizer_sampling = ift.NewtonCG(ic_sampling_nl)

        # Set up likelihood energy and information Hamiltonian
        likelihood_energy = (ift.GaussianEnergy(data, inverse_covariance=N.inverse) @ signal_response)
        H = ift.StandardHamiltonian(likelihood_energy, ic_sampling)


        initial_mean = ift.MultiField.full(H.domain, 0.)
        mean = initial_mean

        plot = ift.Plot()
        plot.add(ground_truth, title='Ground Truth', zmin=0, zmax=1)
        plot.add(R.adjoint_times(data), title='Data')
        plot.add([pspec.force(mean)], title='Power Spectrum')
        plot.output(ny=1, nx=3, xsize=24, ysize=6, name=filename.format("setup"))

        # number of samples used to estimate the KL
        # Minimize KL
        n_iterations = 6
        n_samples = lambda iiter: 10 if iiter < 5 else 20
        samples = ift.optimize_kl(likelihood_energy, n_iterations, n_samples,
                                  minimizer, ic_sampling, minimizer_sampling,
                                  plottable_operators={"signal": (signal, dict(vmin=0, vmax=1)),
                                                       "power spectrum": pspec},
                                  ground_truth_position=None,
                                  output_directory="output_ift_inversion",
                                  overwrite=True, comm=comm)

        if True:
            # Load result from disk. May be useful for long inference runs, where
            # inference and posterior analysis are split into two steps
            samples = ift.ResidualSampleList.load("output_ift_inversion/pickle/last", comm=comm)

        # Plotting
        filename_res = filename.format("results")
        plot = ift.Plot()
        mean, var = samples.sample_stat(signal)
        plot.add(mean, title="Posterior Mean", vmin=0, vmax=1)
        plot.add(var.sqrt(), title="Posterior Standard Deviation", vmin=0)

        nsamples = samples.n_samples
        logspec = pspec.log()
        #plot.add(list(samples.iterator(pspec)) +
        #         [pspec.force(ground_truth), samples.average(logspec).exp()],
        #         title="Sampled Posterior Power Spectrum",
        #         linewidth=[1.] * nsamples + [3., 3.],
        #         label=[None] * nsamples + ['Ground truth', 'Posterior mean'])
        return mean.val.T, var.sqrt().val.T















