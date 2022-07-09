import Line
#from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from Line import Line
import math as m
import sys
import numpy as np
import nifty7 as ift
import plotly.graph_objects as go


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
                line_points = Line(Reflector.position-DOAS_device.position, DOAS_device.position).get_plot_points()
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




    def measure(self):
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



        def line_eq(t, position, orientation):
            alpha, beta, gamma = orientation
            rot_X_Axis = np.array([[1, 0, 0], [0, m.cos(alpha), -m.sin(alpha)], [0, m.sin(alpha), m.cos(alpha)]])
            rot_Y_Axis = np.array([[m.cos(beta), 0, m.sin(beta)], [0, 1, 0], [-m.sin(beta), 0, m.cos(beta)]])
            rot_Z_Axis = np.array([[m.cos(gamma), -m.sin(gamma), 0], [m.sin(gamma), m.cos(gamma), 0], [0, 0, 1]])
            R_C = rot_X_Axis.dot(rot_Y_Axis).dot(rot_Z_Axis)
            return position + np.array([0,0,-1]).dot(R_C)*t
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
            #print(coordinates)

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


    def measurement_inversion_2D(self):
        #b = self.measurement
        #x = [array of gaussian params] -> position of max, sigmax, sigmay
        #A = Verknüpft x und b -> A*x = b
        # also muss in A die Richtung der linie und die Gaussverteilung kodiert sein also quasi ein linienintegral entlang einer 2D gauss-verteilung

        def c(h, c1, c2, sigmax, sigmay):
            return c1 * np.exp(-h * c2)

        def diff(x):
            pass

    def ift_inversion_approach(self):

        def build_LOSs(doas_pos, refl_pos, normalization):
            LOS_starts = [[], []]
            LOS_ends = [[], []]
            for doas in doas_pos:
                for i in range(len(refl_pos)):
                    LOS_starts[0].append(doas[0]/normalization)
                    LOS_starts[1].append(doas[1]/normalization)
                    LOS_ends[0].append(refl_pos[i][0]/normalization)
                    LOS_ends[1].append(refl_pos[i][1]/normalization)
            return LOS_starts, LOS_ends

        filename = "testing_field_inversion_{}.png"
        position_space = ift.RGSpace((self.simulated_field.shape[0], self.simulated_field.shape[1]), distances=(1, 1))
        normalized_field=(np.sum(self.simulated_field, axis=2)/ np.max(np.sum(self.simulated_field, axis=2)))

        #  For a detailed showcase of the effects the parameters
        #  of the CorrelatedField model have on the generated fields,
        #  see 'getting_started_4_CorrelatedFields.ipynb'.

        args = {
            'offset_mean': 0,
            'offset_std': (1e-3, 1e-6),
            # Amplitude of field fluctuations
            'fluctuations': (1., 0.8),  # 1.0, 1e-2
            # Exponent of power law power spectrum component
            'loglogavgslope': (-3., 1),  # -6.0, 1
            # Amplitude of integrated Wiener process power spectrum component
            'flexibility': (2, 1.),  # 1.0, 0.5
            # How ragged the integrated Wiener process component is
            'asperity': (0.5, 0.4)  # 0.1, 0.5
        }

        correlated_field = ift.SimpleCorrelatedField(position_space, **args)
        pspec = correlated_field.power_spectrum

        # Apply a nonlinearity
        signal = ift.sigmoid(correlated_field) #das hier sollte ich ändern!

        # Build the line-of-sight response and define signal response
        doas_pos = self.return_positions(dim="2D")[0]
        refl_pos = self.return_positions(dim="2D")[1]
        LOS_starts, LOS_ends = build_LOSs(doas_pos, refl_pos, 1)
        R = ift.LOSResponse(position_space, starts=LOS_starts, ends=LOS_ends)
        signal_response = R(signal)

        # Specify noise
        data_space = R.target
        noise = .001
        N = ift.ScalingOperator(data_space, noise)

        # Generate mock signal and data
        # Ich möchte normalized_field als mock field benutzen.
        #mock_position = ift.from_random(signal_response.domain, 'normal')
        #mock_position = ift.Field(ift.DomainTuple.make(position_space), val=normalized_field)
        mock_position = ift.MultiField(signal_response.domain, val=normalized_field)
        #mock_position = ift.makeField(signal_response.domain, {"asperity:": (0.5,0.4) ,"bar": normalized_field})

        plot = ift.Plot()
        plot.add(signal(mock_position), title='Ground Truth', zmin=0, zmax=1)
        plot.output()

        data = mock_position + N.draw_sample_with_dtype(dtype=np.float64)

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
        likelihood_energy = (ift.GaussianEnergy(mean=data, inverse_covariance=N.inverse) @ signal_response)
        H = ift.StandardHamiltonian(likelihood_energy, ic_sampling)

        initial_mean = ift.MultiField.full(H.domain, 0.)
        mean = initial_mean

        plot = ift.Plot()
        plot.add(signal(mock_position), title='Ground Truth', zmin=0, zmax=1)
        plot.add(R.adjoint_times(data), title='Data')
        plot.add([pspec.force(mock_position)], title='Power Spectrum')
        plot.output(ny=1, nx=3, xsize=24, ysize=6, name=filename.format("setup"))

        # number of samples used to estimate the KL
        N_samples = 10

        # Draw new samples to approximate the KL six times
        for i in range(6):
            if i == 5:
                # Double the number of samples in the last step for better statistics
                N_samples = 2 * N_samples
            # Draw new samples and minimize KL
            KL = ift.GeoMetricKL(mean, H, N_samples, minimizer_sampling, True)
            KL, convergence = minimizer(KL)
            mean = KL.position
            ift.extra.minisanity(data, lambda x: N.inverse, signal_response,
                                 KL.position, KL.samples)

            # Plot current reconstruction
            plot = ift.Plot()
            plot.add(signal(KL.position), title="Latent mean", zmin=0, zmax=1)
            plot.add([pspec.force(KL.position + ss) for ss in KL.samples],
                     title="Samples power spectrum")
            plot.output(ny=1, ysize=6, xsize=16,
                        name=filename.format("loop_{:02d}".format(i)))

        sc = ift.StatCalculator()
        for sample in KL.samples:
            sc.add(signal(sample + KL.position))

        # Plotting
        filename_res = filename.format("results")
        plot = ift.Plot()
        plot.add(sc.mean, title="Posterior Mean", zmin=0, zmax=1)
        plot.add(ift.sqrt(sc.var), title="Posterior Standard Deviation")

        powers = [pspec.force(s + KL.position) for s in KL.samples]
        sc = ift.StatCalculator()
        for pp in powers:
            sc.add(pp.log())
        plot.add(
            powers + [pspec.force(mock_position),
                      pspec.force(KL.position), sc.mean.exp()],
            title="Sampled Posterior Power Spectrum",
            linewidth=[1.] * len(powers) + [3., 3., 3.],
            label=[None] * len(powers) + ['Ground truth', 'Posterior latent mean', 'Posterior mean'])
        plot.output(ny=1, nx=3, xsize=24, ysize=6, name=filename_res)
        print("Saved results as '{}'.".format(filename_res))

























