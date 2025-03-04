import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns

class MappingType(Enum):
    Identity = 0
    Linear = 1
    Affine = 2

class ODESolver(Enum):
    SemiImplicit = 0
    Explicit = 1
    RungeKutta = 2

class WormnetCell(nn.Module):
    def __init__(self, wormnet_architecture):
        super(WormnetCell, self).__init__()
        self._architecture = wormnet_architecture
        self._input_size = -1
        self._num_units = wormnet_architecture._num_units
        self._output_size = wormnet_architecture._output_size
        self._is_built = False
        self._implicit_param_constraints = False
        self._ode_solver_unfolds = 6
        self._solver = ODESolver.SemiImplicit
        self._input_mapping = MappingType.Affine
        self._output_mapping = MappingType.Affine
        self._erev_init_factor = torch.tensor(1, device=0)
        self._implict_constraints = False
        self._w_init_max = torch.tensor(1.0, device=0)
        self._w_init_min = torch.tensor(0.1, device=0)
        self._cm_init_min = torch.tensor(0.5, device=0)
        self._cm_init_max = torch.tensor(0.5, device=0)
        self._gleak_init_min = torch.tensor(1, device=0)
        self._gleak_init_max = torch.tensor(1, device=0)
        self._w_min_value = torch.tensor(0.001, device=0)
        self._w_max_value = torch.tensor(100, device=0)
        self._gleak_min_value = torch.tensor(0.001, device=0)
        self._gleak_max_value = torch.tensor(100, device=0)
        self._cm_t_min_value = torch.tensor(0.0001, device=0)
        self._cm_t_max_value = torch.tensor(1000, device=0)
        self._fix_cm = None
        self._fix_gleak = None
        self._fix_vleak = None

        self._input_size = 32  # 32
        _input_size, _num_units, _output_size, _sensory_adjacency_matrix, _adjacency_matrix = self._architecture.get_graph(self._input_size)
        assert _input_size == self._input_size
        assert _num_units == self._num_units
        assert _output_size == self._output_size
        self._sensory_adjacency_matrix = torch.tensor(_sensory_adjacency_matrix, device=0)
        self._adjacency_matrix = torch.tensor(_adjacency_matrix, device=0)
        self._get_variables()


    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._output_size

    def _map_inputs(self, inputs, reuse_scope=False):
        if self._input_mapping == MappingType.Affine or self._input_mapping == MappingType.Linear:
            w = nn.Parameter(torch.ones(self._input_size, device=inputs.device))
            inputs = inputs * w
        if self._input_mapping == MappingType.Affine:
            b = nn.Parameter(torch.zeros(self._input_size, device=inputs.device))
            inputs = inputs + b
        return inputs

    def _map_outputs(self, states):
        motor_neurons = states[:, :self._output_size]
        if self._output_mapping == MappingType.Affine or self._output_mapping == MappingType.Linear:
            w = nn.Parameter(torch.ones(self._output_size, device=states.device))
            motor_neurons = motor_neurons * w
        if self._output_mapping == MappingType.Affine:
            b = nn.Parameter(torch.zeros(self._output_size, device=states.device))
            motor_neurons = motor_neurons + b
        return motor_neurons

    def count_params(self):
        num_of_synapses = int(np.sum(np.abs(self._adjacency_matrix)))
        num_of_sensory_synapses = int(np.sum(np.abs(self._sensory_adjacency_matrix)))
        total_parameters = 0
        if self._fix_cm is None:
            total_parameters += self._num_units
        if self._fix_gleak is None:
            total_parameters += self._num_units
        if self._fix_vleak is None:
            total_parameters += self._num_units
        total_parameters += 4 * (num_of_sensory_synapses + num_of_synapses)
        return total_parameters

    def build(self, input_shape):
        pass

    def forward(self, inputs, state, b=None, s=None):
        inputs = self._map_inputs(inputs)

        if self._solver == ODESolver.Explicit:
            next_state = self._ode_step_explicit(inputs, state, _ode_solver_unfolds=self._ode_solver_unfolds)
        elif self._solver == ODESolver.SemiImplicit:
            next_state = self._ode_step(inputs, state)
        elif self._solver == ODESolver.RungeKutta:
            next_state = self._ode_step_runge_kutta(inputs, state)
        else:
            raise ValueError("Unknown ODE solver '{}'".format(str(self._solver)))

        outputs = self._map_outputs(next_state)
        outputs = outputs.reshape(b, s, 1)
        return outputs, next_state

    def _get_variables(self, device=0):
        self.sensory_mu = nn.Parameter(torch.rand(self._input_size, self._num_units) * 0.5 + 0.3)
        self.sensory_sigma = nn.Parameter(torch.rand(self._input_size, self._num_units) * 5.0 + 3.0)  # [32, 19]
        self.sensory_W = nn.Parameter(torch.abs(self._sensory_adjacency_matrix) * torch.rand(self._input_size, self._num_units, device=0) * (self._w_init_max - self._w_init_min) + self._w_init_min)
        self.sensory_erev = nn.Parameter(self._sensory_adjacency_matrix * self._erev_init_factor)

        self.mu = nn.Parameter(torch.rand(self._num_units, self._num_units) * 0.5 + 0.3)
        self.sigma = nn.Parameter(torch.rand(self._num_units, self._num_units) * 5.0 + 3.0)
        self.W = nn.Parameter(torch.abs(self._adjacency_matrix) * torch.rand(self._num_units, self._num_units, device=0) * (self._w_init_max - self._w_init_min) + self._w_init_min)
        self.erev = nn.Parameter(self._adjacency_matrix * self._erev_init_factor)

        # import torch.nn.init as init
        # init.xavier_uniform_(self.sensory_mu)
        # init.xavier_uniform_(self.sensory_sigma)
        # init.xavier_uniform_(self.sensory_W)
        # init.xavier_uniform_(self.sensory_erev)
        #
        # init.xavier_uniform_(self.mu)
        # init.xavier_uniform_(self.sigma)
        # init.xavier_uniform_(self.W)
        # init.xavier_uniform_(self.erev)

        if self._fix_vleak is None:
            self.vleak = nn.Parameter(torch.rand(self._num_units) * 0.4 - 0.2)
        else:
            self.vleak = nn.Parameter(torch.tensor(self._fix_vleak), requires_grad=False)

        if self._fix_gleak is None:
            if self._gleak_init_max > self._gleak_init_min:
                self.gleak = nn.Parameter(torch.rand(self._num_units) * (self._gleak_init_max - self._gleak_init_min) + self._gleak_init_min)
            else:
                self.gleak = nn.Parameter(torch.ones(self._num_units, device=0) * self._gleak_init_min)
        else:
            self.gleak = nn.Parameter(torch.tensor(self._fix_gleak), requires_grad=False)

        if self._fix_cm is None:
            if self._cm_init_max > self._cm_init_min:
                self.cm_t = nn.Parameter(torch.rand(self._num_units) * (self._cm_init_max - self._cm_init_min) + self._cm_init_min)
            else:
                self.cm_t = nn.Parameter(torch.ones(self._num_units, device=0) * self._cm_init_min)
        else:
            self.cm_t = nn.Parameter(torch.tensor(self._fix_cm), requires_grad=False)

        if self._implicit_param_constraints:
            self.W = nn.functional.softplus(self.W)
            self.sensory_W = nn.functional.softplus(self.sensory_W)
            self.gleak = nn.functional.softplus(self.gleak)
            self.cm_t = nn.functional.softplus(self.cm_t)

        self._sensory_synapse_mask = torch.abs(self._sensory_adjacency_matrix)
        self._synapse_mask = torch.abs(self._adjacency_matrix)

    def _ode_step(self, inputs, state):
        v_pre = state
        sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        if self._implicit_param_constraints:
            sensory_w_activation *= self._sensory_synapse_mask
        sensory_rev_activation = sensory_w_activation * self.sensory_erev
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        for t in range(self._ode_solver_unfolds):
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)
            if self._implicit_param_constraints:
                w_activation *= self._synapse_mask
            rev_activation = w_activation * self.erev

            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory  # rev_activation [16, 19, 19]  # w_numerator_sensory [16, 19]
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory

            numerator = self.cm_t * v_pre + self.gleak * self.vleak + w_numerator
            denominator = self.cm_t + self.gleak + w_denominator

            v_pre = numerator / denominator

        return v_pre

    def _f_prime(self, inputs, state):
        v_pre = state
        sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        if self._implicit_param_constraints:
            sensory_w_activation *= self._sensory_synapse_mask
        w_reduced_sensory = torch.sum(sensory_w_activation, dim=1)

        w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)
        if self._implicit_param_constraints:
            w_activation *= self._synapse_mask
        w_reduced_synapse = torch.sum(w_activation, dim=1)

        sensory_in = self.sensory_erev * sensory_w_activation
        synapse_in = self.erev * w_activation

        sum_in = torch.sum(sensory_in, dim=1) - v_pre * w_reduced_synapse + torch.sum(synapse_in, dim=1) - v_pre * w_reduced_sensory

        f_prime = 1 / self.cm_t * (self.gleak * (self.vleak - v_pre) + sum_in)

        return f_prime

    def _ode_step_runge_kutta(self, inputs, state):
        h = 0.1
        for i in range(self._ode_solver_unfolds):
            k1 = h * self._f_prime(inputs, state)
            k2 = h * self._f_prime(inputs, state + k1 * 0.5)
            k3 = h * self._f_prime(inputs, state + k2 * 0.5)
            k4 = h * self._f_prime(inputs, state + k3)

            state = state + 1.0 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return state

    def _ode_step_explicit(self, inputs, state, _ode_solver_unfolds):
        v_pre = state
        sensory_w_activation = self.sensory_W * self._sigmoid(inputs, self.sensory_mu, self.sensory_sigma)
        if self._implicit_param_constraints:
            sensory_w_activation *= self._sensory_synapse_mask
        w_reduced_sensory = torch.sum(sensory_w_activation, dim=1)

        for t in range(_ode_solver_unfolds):
            w_activation = self.W * self._sigmoid(v_pre, self.mu, self.sigma)
            if self._implicit_param_constraints:
                w_activation *= self._synapse_mask
            w_reduced_synapse = torch.sum(w_activation, dim=1)

            sensory_in = self.sensory_erev * sensory_w_activation
            synapse_in = self.erev * w_activation

            sum_in = torch.sum(sensory_in, dim=1) - v_pre * w_reduced_synapse + torch.sum(synapse_in, dim=1) - v_pre * w_reduced_sensory

            f_prime = 1 / self.cm_t * (self.gleak * (self.vleak - v_pre) + sum_in)

            v_pre = v_pre + 0.1 * f_prime

        return v_pre

    def _sigmoid(self, v_pre, mu, sigma):
        v_pre = v_pre.view(-1, v_pre.shape[-1], 1)
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def summary(self):
        print("=== Network statistics ===")
        print("# Neurons : " + str(self._num_units))

        num_of_synapses = int(np.sum(np.abs(self._adjacency_matrix)))
        num_of_sensory_synapses = int(np.sum(np.abs(self._sensory_adjacency_matrix)))

        print("# Synapses: " + str(num_of_synapses + num_of_sensory_synapses) + " (" + str(num_of_sensory_synapses) + "/" + str(num_of_synapses) + ")")
        print("# Inputs  : " + str(self._input_size))
        print("# Outputs : " + str(self._output_size))

        total_parameters = 0
        if self._fix_cm is None:
            total_parameters += self._num_units
        if self._fix_gleak is None:
            total_parameters += self._num_units
        if self._fix_vleak is None:
            total_parameters += self._num_units

        total_parameters += 4 * (num_of_sensory_synapses + num_of_synapses)

        print("# Parameters: " + str(total_parameters))

    def export_parameters(self, path):
        if not path.endswith(".npz") and not os.path.exists(path):
            os.makedirs(path)

        gleak = self.gleak.detach().cpu().numpy()
        vleak = self.vleak.detach().cpu().numpy()
        cm = self.cm_t.detach().cpu().numpy()
        sigma = self.sigma.detach().cpu().numpy()
        mu = self.mu.detach().cpu().numpy()
        W = self.W.detach().cpu().numpy()
        erev = self.erev.detach().cpu().numpy()
        sensory_sigma = self.sensory_sigma.detach().cpu().numpy()
        sensory_mu = self.sensory_mu.detach().cpu().numpy()
        sensory_W = self.sensory_W.detach().cpu().numpy()
        sensory_erev = self.sensory_erev.detach().cpu().numpy()

        if path.endswith(".npz"):
            np.savez(
                path,
                gleak=gleak,
                vleak=vleak,
                cm=cm,
                sigma=sigma,
                mu=mu,
                W=W,
                erev=erev,
                sensory_sigma=sensory_sigma,
                sensory_mu=sensory_mu,
                sensory_W=sensory_W,
                sensory_erev=sensory_erev,
                adjacency_matrix=self._adjacency_matrix,
                sensory_adjacency_matrix=self._sensory_adjacency_matrix
            )
            return

        with open(os.path.join(path, 'neuron_parameters.csv'), 'w') as f:
            f.write("gleak;vleak;cm\n")
            for i in range(self._num_units):
                f.write("{};{};{}\n".format(gleak[i], vleak[i], cm[i]))

        with open(os.path.join(path, 'sensory_synapses.csv'), 'w') as f:
            f.write("src;dest;sigma;mu,w;erev;erev_init\n")
            for src in range(self._input_size):
                for dest in range(self._num_units):
                    if self._sensory_adjacency_matrix[src, dest] != 0:
                        f.write("{:d};{:d};{};{};{};{};{}\n".format(
                            src,
                            dest,
                            sensory_sigma[src, dest],
                            sensory_mu[src, dest],
                            sensory_W[src, dest],
                            sensory_erev[src, dest],
                            self._sensory_adjacency_matrix[src, dest],
                        ))

        with open(os.path.join(path, 'inter_synapses.csv'), 'w') as f:
            f.write("src;dest;sigma;mu,w;erev;erev_init\n")
            for src in range(self._num_units):
                for dest in range(self._num_units):
                    if self._adjacency_matrix[src, dest] != 0:
                        f.write("{:d};{:d};{};{};{};{};{}\n".format(
                            src,
                            dest,
                            sigma[src, dest],
                            mu[src, dest],
                            W[src, dest],
                            erev[src, dest],
                            self._adjacency_matrix[src, dest],
                        ))

    def get_param_constrain_op(self):
        if self._implicit_param_constraints:
            return None

        self.cm_t.data = torch.clamp(self.cm_t, self._cm_t_min_value, self._cm_t_max_value)
        self.gleak.data = torch.clamp(self.gleak, self._gleak_min_value, self._gleak_max_value)

        _synapse_mask = torch.abs(self._adjacency_matrix)
        _sensory_synapse_mask = torch.abs(self._sensory_adjacency_matrix)

        self.W.data = torch.clamp(self.W * _synapse_mask, self._w_min_value, self._w_max_value)
        self.sensory_W.data = torch.clamp(self.sensory_W * _sensory_synapse_mask, self._w_min_value, self._w_max_value)

class WormnetArchitecture:
    def __init__(self, output_size, num_units, input_size=None):
        self._output_size = output_size
        self._num_units = num_units
        self._adjacency_matrix = np.zeros([self._num_units, self._num_units], dtype=np.float32)
        self._input_size = input_size
        if self._input_size is not None:
            self._sensory_adjacency_matrix = np.zeros([self._input_size, self._num_units], dtype=np.float32)

    def add_synapse(self, src, dest, polarity):
        if src < 0 or src >= self._num_units:
            raise ValueError("Source of synapse ({},{}) out of valid range (0,{})".format(src, dest, self._num_units))
        if dest < 0 or dest >= self._num_units:
            raise ValueError("Destination of synapse ({},{}) out of valid range (0,{})".format(src, dest, self._num_units))
        self._adjacency_matrix[src, dest] = polarity

    def add_sensory_synapse(self, src, dest, polarity):
        if self._input_size is None:
            raise ValueError("Input size must be defined before sensory synapses can be added")
        if src < 0 or src >= self._input_size:
            raise ValueError("Source of sensory synapse ({},{}) out of valid range (0,{})".format(src, dest, self._num_units))
        if dest < 0 or dest >= self._num_units:
            raise ValueError("Destination of sensory synapse ({},{}) out of valid range (0,{})".format(src, dest, self._num_units))
        self._sensory_adjacency_matrix[src, dest] = polarity

    def construct_graph(self):
        pass

    def get_graph(self, input_size):
        if self._input_size is not None and self._input_size != input_size:
            raise ValueError("Input size of wormnet architecture was set to {} in constructor call, and now {} was given as input size".format(self._input_size, input_size))

        if self._input_size is None:
            self._input_size = input_size
            self._sensory_adjacency_matrix = np.zeros([self._input_size, self._num_units], dtype=np.float32)

        self.construct_graph()

        return (self._input_size, self._num_units, self._output_size, self._sensory_adjacency_matrix, self._adjacency_matrix)

class FullyConnectedWormnetArchitecture(WormnetArchitecture):
    def construct_graph(self):
        self._rng = np.random.RandomState(2018123)
        for src in range(self._input_size):
            for dest in range(self._num_units):
                polarity = 1
                if self._rng.rand() > 0.5:
                    polarity = -1
                self.add_sensory_synapse(src, dest, polarity)
        for src in range(self._num_units):
            for dest in range(self._num_units):
                if src == dest:
                    continue
                polarity = 1
                if self._rng.rand() > 0.5:
                    polarity = -1
                self.add_synapse(src, dest, polarity)

class RandomWormnetArchitecture(WormnetArchitecture):
    def __init__(self, output_size, num_units, sensory_density=3, inter_density=8, motor_density=4, seed=20190120, input_size=None):
        super().__init__(output_size, num_units, input_size)
        self._sensory_density = sensory_density
        self._inter_density = inter_density
        self._motor_density = motor_density
        self._seed = seed

    def construct_graph(self):
        self._rng = np.random.RandomState(self._seed)

        for src in range(self._input_size):
            dest = self._rng.permutation(np.arange(0, self._num_units))
            for i in range(self._sensory_density):
                polarity = 1
                if self._rng.rand() > 0.5:
                    polarity = -1
                self.add_sensory_synapse(src, dest[i], polarity)

        for dest in range(self._output_size):
            src = self._rng.permutation(np.arange(0, self._num_units))
            for i in range(self._motor_density):
                polarity = 1
                if self._rng.rand() > 0.5:
                    polarity = -1
                self.add_synapse(src[i], dest, polarity)

        inter_neuron_range = np.arange(self._output_size, self._num_units)
        src_index, dest_index = np.meshgrid(inter_neuron_range, inter_neuron_range)
        src_index = src_index.flatten()
        dest_index = dest_index.flatten()

        index_shuffle = self._rng.permutation(src_index.shape[0])
        src_permutation = src_index[index_shuffle]
        dest_permutation = dest_index[index_shuffle]

        for i in range(int(np.max([self._inter_density, src_index.shape[0]]))):
            polarity = 1
            if self._rng.rand() > 0.5:
                polarity = -1
            self.add_synapse(src_permutation[i], dest_permutation[i], polarity)

        self._forward_reachablity_analysis()
        self._backward_reachablity_analysis()

    def _forward_reachablity_analysis(self):
        forward_unreachable = list(range(self._num_units))
        forward_reachable = []

        for src in range(self._input_size):
            for dest in range(self._num_units):
                if self._sensory_adjacency_matrix[src, dest] != 0:
                    if dest in forward_unreachable:
                        forward_unreachable.remove(dest)
                        forward_reachable.append(dest)

        reachable_count = 0
        while reachable_count != len(forward_reachable):
            reachable_count = len(forward_reachable)
            for src in range(self._num_units):
                for dest in range(self._num_units):
                    if self._adjacency_matrix[src, dest] != 0:
                        if dest in forward_unreachable:
                            forward_unreachable.remove(dest)
                            forward_reachable.append(dest)

        while len(forward_unreachable) > 0:
            dest = forward_unreachable.pop(0)
            shuffle = self._rng.permutation(np.arange(0, len(forward_reachable)))
            for i in range(3):
                polarity = 1
                if self._rng.rand() > 0.5:
                    polarity = -1
                self.add_synapse(forward_reachable[shuffle[i]], dest, polarity)

    def _backward_reachablity_analysis(self):
        backward_unreachable = list(range(self._num_units))
        backward_reachable = []

        for i in range(self._output_size):
            backward_unreachable.remove(i)
            backward_reachable.append(i)

        reachable_count = 0
        while reachable_count != len(backward_reachable):
            reachable_count = len(backward_reachable)
            for src in range(self._num_units):
                for dest in range(self._num_units):
                    if self._adjacency_matrix[src, dest] != 0:
                        if src in backward_unreachable:
                            backward_unreachable.remove(src)
                            backward_reachable.append(src)

        while len(backward_unreachable) > 0:
            src = backward_unreachable.pop(0)
            shuffle = self._rng.permutation(np.arange(0, len(backward_reachable)))
            for i in range(3):
                polarity = 1
                if self._rng.rand() > 0.5:
                    polarity = -1
                self.add_synapse(src, backward_reachable[shuffle[i]], polarity)

class CommandLayerWormnetArchitectureMK2(WormnetArchitecture):
    def __init__(self, output_size, num_interneurons=6, num_command_neurons=4, sensory_density=4, inter_density=3, recurrency=2, motor_density=4, seed=20190120, input_size=None):
        super().__init__(output_size, num_interneurons + num_command_neurons + output_size, input_size)
        self._num_interneurons = num_interneurons
        self._num_command_neurons = num_command_neurons
        self._sensory_density = sensory_density
        self._inter_density = inter_density
        self._recurrency = recurrency
        self._motor_density = motor_density
        self._seed = seed
        self._verbose = False
        self._prob_excitatory = 0.7

        if self._motor_density > self._num_command_neurons:
            raise ValueError("Motor density must be less or equal than the number of command neurons")
        if self._sensory_density > self._num_interneurons:
            raise ValueError("Sensory density must be less or equal than the number of interneurons neurons")
        if self._inter_density > self._num_command_neurons:
            raise ValueError("Inter density must be less or equal than the number of command neurons")

    def _connect_sensory_inter(self, src, dest, polarity=None):
        if polarity is None:
            polarity = 1
            if self._rng.rand() > self._prob_excitatory:
                polarity = -1
        real_dest = dest + self._output_size + self._num_command_neurons
        real_src = src
        if self._verbose:
            print("Senosry ({}) -> Inter ({}) [{} -> {}]".format(src, dest, real_src, real_dest))
        self.add_sensory_synapse(real_src, real_dest, polarity)

    def _connect_inter_command(self, src, dest, polarity=None):
        if polarity is None:
            polarity = 1
            if self._rng.rand() > self._prob_excitatory:
                polarity = -1
        real_dest = dest + self._output_size
        real_src = src + self._output_size + self._num_command_neurons
        if self._verbose:
            print("Inter ({}) -> Command ({}) [{} -> {}]".format(src, dest, real_src, real_dest))
        self.add_synapse(real_src, real_dest, polarity)

    def _connect_command_command(self, src, dest, polarity=None):
        if polarity is None:
            polarity = 1
            if self._rng.rand() > self._prob_excitatory:
                polarity = -1
        real_dest = dest + self._output_size
        real_src = src + self._output_size
        if self._verbose:
            print("Command ({}) -> Command ({}) [{} -> {}]".format(src, dest, real_src, real_dest))
        self.add_synapse(real_src, real_dest, polarity)

    def _connect_command_motor(self, src, dest, polarity=None):
        if polarity is None:
            polarity = 1
            if self._rng.rand() > self._prob_excitatory:
                polarity = -1
        real_dest = dest
        real_src = src + self._output_size
        if self._verbose:
            print("Command ({}) -> Motor ({}) [{} -> {}]".format(src, dest, real_src, real_dest))
        self.add_synapse(real_src, real_dest, polarity)

    def construct_graph(self):
        self._rng = np.random.RandomState(self._seed)

        unreachable_interneurons = list(range(self._num_interneurons))
        for src in range(self._input_size):
            dest_index = self._rng.permutation(np.arange(0, self._num_interneurons))
            for i in range(self._sensory_density):
                if dest_index[i] in unreachable_interneurons:
                    unreachable_interneurons.remove(dest_index[i])
                self._connect_sensory_inter(src, dest_index[i])

        mean_interneuron_fanin = int(self._input_size * self._sensory_density / self._num_interneurons)
        for i in unreachable_interneurons:
            src = self._rng.permutation(np.arange(0, self._input_size))
            for j in range(mean_interneuron_fanin):
                self._connect_sensory_inter(src[j], i)

        unreachable_commandneurons = list(range(self._num_command_neurons))
        for src_index in range(self._num_interneurons):
            dest_index = self._rng.permutation(np.arange(0, self._num_command_neurons))
            for i in range(self._inter_density):
                if dest_index[i] in unreachable_commandneurons:
                    unreachable_commandneurons.remove(dest_index[i])
                self._connect_inter_command(src_index, dest_index[i])

        mean_commandneurons_fanin = int(self._num_interneurons * self._inter_density / self._num_command_neurons)
        for i in unreachable_commandneurons:
            src_index = self._rng.permutation(np.arange(0, self._num_interneurons))
            for j in range(mean_commandneurons_fanin):
                self._connect_inter_command(src_index[j], i)

        for i in range(self._recurrency):
            src = self._rng.randint(0, self._num_command_neurons)
            dest = self._rng.randint(0, self._num_command_neurons)
            self._connect_command_command(src, dest)

        unreachable_commandneurons = list(range(self._num_command_neurons))
        for dest in range(self._output_size):
            src_index = self._rng.permutation(np.arange(0, self._num_command_neurons))
            for i in range(self._motor_density):
                if src_index[i] in unreachable_commandneurons:
                    unreachable_commandneurons.remove(src_index[i])
                self._connect_command_motor(src_index[i], dest)

        mean_motorneuron_fanin = int(self._output_size * self._motor_density / self._num_command_neurons)
        for i in unreachable_commandneurons:
            dest = self._rng.permutation(np.arange(0, self._output_size))
            for j in range(mean_motorneuron_fanin):
                self._connect_command_motor(i, dest[j])

class CommandLayerWormnetArchitecture(WormnetArchitecture):
    def __init__(self, output_size, num_interneurons=6, num_command_neurons=4, sensory_density=4, inter_density=2, recurrency=4, motor_density=4, seed=20190120, input_size=None):
        super().__init__(output_size, num_interneurons + num_command_neurons + output_size, input_size)
        self._num_interneurons = num_command_neurons
        self._num_command_neurons = num_interneurons
        self._sensory_density = sensory_density
        self._inter_density = inter_density
        self._recurrency = recurrency
        self._motor_density = motor_density
        self._seed = seed
        self._verbose = False
        self._prob_excitatory = 0.7

        if self._motor_density > self._num_command_neurons:
            raise ValueError("Motor density must be less or equal than the number of command neurons")
        if self._sensory_density > self._num_interneurons:
            raise ValueError("Sensory density must be less or equal than the number of interneurons neurons")
        if self._inter_density > self._num_command_neurons:
            raise ValueError("Inter density must be less or equal than the number of command neurons")

    def _connect_sensory_inter(self, src, dest, polarity=None):
        if polarity is None:
            polarity = 1
            if self._rng.rand() > self._prob_excitatory:
                polarity = -1
        real_dest = dest + self._output_size + self._num_command_neurons
        real_src = src
        if self._verbose:
            print("Senosry ({}) -> Inter ({}) [{} -> {}]".format(src, dest, real_src, real_dest))
        self.add_sensory_synapse(real_src, real_dest, polarity)

    def _connect_inter_command(self, src, dest, polarity=None):
        if polarity is None:
            polarity = 1
            if self._rng.rand() > self._prob_excitatory:
                polarity = -1
        real_dest = dest + self._output_size
        real_src = src + self._output_size + self._num_command_neurons
        if self._verbose:
            print("Inter ({}) -> Command ({}) [{} -> {}]".format(src, dest, real_src, real_dest))
        self.add_synapse(real_src, real_dest, polarity)

    def _connect_command_command(self, src, dest, polarity=None):
        if polarity is None:
            polarity = 1
            if self._rng.rand() > self._prob_excitatory:
                polarity = -1
        real_dest = dest + self._output_size
        real_src = src + self._output_size
        if self._verbose:
            print("Command ({}) -> Command ({}) [{} -> {}]".format(src, dest, real_src, real_dest))
        self.add_synapse(real_src, real_dest, polarity)

    def _connect_command_motor(self, src, dest, polarity=None):
        if polarity is None:
            polarity = 1
            if self._rng.rand() > self._prob_excitatory:
                polarity = -1
        real_dest = dest
        real_src = src + self._output_size
        if self._verbose:
            print("Command ({}) -> Motor ({}) [{} -> {}]".format(src, dest, real_src, real_dest))
        self.add_synapse(real_src, real_dest, polarity)

    def construct_graph(self):
        self._rng = np.random.RandomState(self._seed)

        unreachable_interneurons = list(range(self._num_interneurons))
        for src in range(self._input_size):
            dest_index = self._rng.permutation(np.arange(0, self._num_interneurons))
            for i in range(self._sensory_density):
                if dest_index[i] in unreachable_interneurons:
                    unreachable_interneurons.remove(dest_index[i])
                self._connect_sensory_inter(src, dest_index[i])

        mean_interneuron_fanin = int(self._input_size * self._sensory_density / self._num_interneurons)
        for i in unreachable_interneurons:
            src = self._rng.permutation(np.arange(0, self._input_size))
            for j in range(mean_interneuron_fanin):
                self._connect_sensory_inter(src[j], i)

        unreachable_commandneurons = list(range(self._num_command_neurons))
        for src_index in range(self._inter_density):
            dest_index = self._rng.permutation(np.arange(0, self._num_command_neurons))
            for i in range(self._inter_density):
                if dest_index[i] in unreachable_commandneurons:
                    unreachable_commandneurons.remove(dest_index[i])
                self._connect_inter_command(src_index, dest_index[i])

        mean_commandneurons_fanin = int(self._num_interneurons * self._inter_density / self._num_command_neurons)
        for i in unreachable_commandneurons:
            src_index = self._rng.permutation(np.arange(0, self._num_interneurons))
            for j in range(mean_commandneurons_fanin):
                self._connect_inter_command(src_index[j], i)

        for i in range(self._recurrency):
            src = self._rng.randint(0, self._num_command_neurons)
            dest = self._rng.randint(0, self._num_command_neurons)
            self._connect_command_command(src, dest)

        unreachable_commandneurons = list(range(self._num_command_neurons))
        for dest in range(self._output_size):
            src_index = self._rng.permutation(np.arange(0, self._num_command_neurons))
            for i in range(self._motor_density):
                if src_index[i] in unreachable_commandneurons:
                    unreachable_commandneurons.remove(src_index[i])
                self._connect_command_motor(src_index[i], dest)

        mean_motorneuron_fanin = int(self._output_size * self._motor_density / self._num_command_neurons)
        for i in unreachable_commandneurons:
            dest = self._rng.permutation(np.arange(0, self._output_size))
            for j in range(mean_motorneuron_fanin):
                self._connect_command_motor(i, dest[j])

if __name__ == '__main__':
    architecture = CommandLayerWormnetArchitectureMK2(1, num_interneurons=8, num_command_neurons=4, sensory_density=4, inter_density=3, recurrency=2, motor_density=4, seed=20190120, input_size=None)
    wm = WormnetCell(architecture)

    time_dimension = 100
    num_of_observations = 2
    batch_size = 1

    train_x = np.array([np.sin(np.linspace(0, 4 * np.pi, time_dimension)), np.cos(np.linspace(0, 4 * np.pi, time_dimension))])
    train_y = np.array(np.sin(np.linspace(0, 8 * np.pi, time_dimension)))

    train_x = np.transpose(train_x, axes=[1, 0]).reshape([time_dimension, 1, 2])
    train_y = train_y.reshape([time_dimension, 1, 1])

    print("Train_x shape: " + str(train_x.shape))
    print("Train_y shape: " + str(train_y.shape))

    build_start = time.time()

    wm_out, _ = wm(torch.tensor(train_x, dtype=torch.float32), torch.zeros(batch_size, wm.state_size))

    ys = torch.tensor(train_y, dtype=torch.float32)
    loss = torch.mean((ys - wm_out) ** 2)

    optimizer = optim.Adam(wm.parameters(), lr=0.001)

    clip_op = wm.get_param_constrain_op()

    build_time = time.time() - build_start
    print("Graph build time: {:0.2f} s".format(build_time))

    max_num_epochs = 401
    train_times = []
    for epoch in range(max_num_epochs):
        start = time.time()
        optimizer.zero_grad()
        wm_out, _ = wm(torch.tensor(train_x, dtype=torch.float32), torch.zeros(batch_size, wm.state_size))
        loss = torch.mean((ys - wm_out) ** 2)
        loss.backward()
        optimizer.step()

        for op in clip_op:
            op()

        if epoch % 20 == 0:
            if not os.path.exists('sine_traces'):
                os.makedirs('sine_traces')

            sns.set()
            plt.figure(figsize=(6, 5))
            plt.plot(wm_out[:, 0, 0].detach().numpy(), label='wormnet')
            plt.plot(train_y[:, 0, 0], linestyle='dashed', label='label signal')
            plt.xlabel('Time steps')
            plt.ylabel('Neuron potential')
            plt.title("Neuron 0")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join('sine_traces', 'epochs_{:04d}.png'.format(epoch)))
            plt.close()

            print('epoch {} Train loss: {:0.2f}'.format(epoch, loss.item()))