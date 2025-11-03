import gymnasium as gym
from gymnasium import spaces
import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import max_len_seq, butter, filtfilt

import numpy as np
# from scipy.integrate import odeint
import scipy.integrate as nt
import matplotlib.pyplot as plt

class oscillator_model_gym:
    def __init__(self, params=None):
        defaults = {
            'w0': 3.0,
            'Omega': 0.3,
            'delta': 0.1,
            'eta': 1.0,
            'nu': 1.0,
            'pi': 0.1,
            'Etil': 0.0,
            'K': -1.1,
            
            'y0': 1e-5 + 1e-5j,
            'T': 800.0,
            'dt': 0.1,
            'prbs_ind': 500,
            'state_dim': 3,
            'control_dim': 2,
            'initial_transition_steps': 1050
        }
        
        # Override defaults with user-specified params
        if params:
            defaults.update(params)
            
        self.__dict__.update(defaults)

        self.wf = self.w0 + self.Omega

        # Time points
        self.y = 0
        self.num_timesteps = int(self.T/self.dt)
        self.t = [0]
        
        #data
        self.u_control = []
        self.prbs_ind = 500
    
    def reset(self, options:dict = None):
        
        self.t = [0]
        self.terminate = False
        self.u_control = []
        self.action = 0.0j
        
        if options is not None and 'state' in options.keys():
            init_state = options['state'][0] + 1j*options['state'][1]
            self.set_initial_condition(init_state)
        else:
            self.set_initial_condition(self.y0)
            for t_transition in range(self.initial_transition_steps):
                # print("###### running transition #####")    
                state = [self.y] if isinstance(self.y,complex) else self.y.y[0]
                self.y = nt.solve_ivp(self.model, [0,self.dt], state, method = 'RK45', t_eval = [self.dt])
        
        return self.complete_state
    
    def set_initial_condition(self, value = None):
        if value:
            self.y = value
        else:
            self.y = (1e-5 + 1e-5j) 
            
    def model(self, t, Btil):
        Et = self.action
        dBtildt = self.eta*self.delta*Btil - self.nu*Btil*abs(Btil)**2 + self.pi*Et + 1j*self.w0*Btil
        return dBtildt
    
    def step(self, action):
        self.action = action
        self.u_control.append(action)

        state = [self.y] if isinstance(self.y,complex) else self.y.y[0]
        self.y = nt.solve_ivp(self.model, [0,self.dt], state, method = 'RK45', t_eval = [self.dt])
        self.t.append(self.t[-1] + self.dt)
        reward = self.get_reward()
        if reward > 6.0:
            self.terminate = True
            
        return self.complete_state, reward, self.terminate, {}

    def get_reward(self):
        return self.complete_state[-1]
     
    def get_y(self):
        return self.y
    
    @property    
    def complete_u(self):
        u  = np.array(self.u_control)[np.newaxis,:]
        cu = np.concatenate((np.real(u),np.imag(u)), axis = 0)
        return cu

    @property
    def complete_state(self, perturb = 0.0):

        y = np.array([self.y])[:,np.newaxis] if isinstance(self.y,complex) else self.y.y
        cs = np.concatenate((np.real(y),np.imag(y)), axis = 0)
        cs = np.concatenate((cs, np.abs(y)**1), axis = 0)
        cs +=perturb
        return cs

class forcings:
    def __init__(self, osgym: oscillator_model_gym):
        self.osgym = osgym
        self.w0 = osgym.w0
        self.Omega = osgym.Omega
        self.wf = osgym.wf
        self.Etil = osgym.Etil
        self.K = osgym.K
        self.dt = osgym.dt
        
        #data
        self.prbs_ind = 500

    def control_forcing(self, x):
        u_control = self.K*x
        self.osgym.u_control.append(u_control)        
        return u_control
    
    # def forcing(self, t):
    #     Et = self.Etil*np.exp(1j*self.wf*t)
    #     self.osgym.u_control.append(Et)
    #     return Et
    
    def prbs(self, amplitude = 1):
        # Generate the PRBS signal
        dt = self.dt #1/30                # Sampling period
        amplitude = amplitude
        N = 16                   # Smaller N for demonstration; adjust as needed
        
        rng = np.random.default_rng()
        state1 = rng.integers(1, 2**N, N)
        state2 = rng.integers(1, 2**N, N)
        real_u = amplitude * (max_len_seq(N, state=state1)[0] * 2 - 1)
        imag_u = amplitude * (max_len_seq(N, state=state2)[0] * 2 - 1)  # PRBS signal
        # u = amplitude * (max_len_seq(N)[0] * 2 - 1)  # PRBS signal


        # Define the desired cutoff frequency for the filter
        fcut = 0.1*self.w0/(2*np.pi)                 # Desired cutoff frequency in Hz
        nyquist = 0.5 / dt       # Nyquist frequency
        normalized_cutoff = fcut / nyquist

        # Design a low-pass Butterworth filter
        order = 4
        b, a = butter(order, normalized_cutoff, btype='low', analog=False)

        # Apply the filter to the PRBS signal
        self.filtered_signal_realu = filtfilt(b, a, real_u)
        self.filtered_signal_imagu = filtfilt(b, a, imag_u)
        # self.filtered_signal = filtfilt(b, a, u)

    
    def forcing(self,t):
        self.prbs_ind +=1
#         print(self.prbs_ind)
#         print(self.filtered_signal[self.prbs_ind])
        # forcing = np.exp(1j*self.w0*t)*self.filtered_signal[self.prbs_ind]
        forcing = (self.filtered_signal_realu[self.prbs_ind] + 1j*self.filtered_signal_imagu[self.prbs_ind])
        return np.array([forcing.real, forcing.imag])

class OscillatorEnv(gym.Env):
    metadata = {'render_modes': []}
    def __init__(self, params: dict = None):
        super().__init__()
        self.osc = oscillator_model_gym(params)
        self.forcings = forcings(self.osc)
        self.forcings.prbs(amplitude = params["amplitude"])
        self.action_space = spaces.Box(low=-2, high=2, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.params = params

    def reset(self, seed=None, options: dict = None):
        super().reset(seed=seed)
        self.forcings.prbs(amplitude = self.params["amplitude"]) 
        if options is not None and 'state' in options.keys():
            state = self.osc.reset(options = {"state": options['state']})
        else:
            state = self.osc.reset()
        obs = np.asarray(state).reshape(-1)  # now (3,)
        return obs, {}

    def step(self, action):
        real, imag = action[0], action[1]
        action_complex = real + 1j * imag
        state, reward, terminated, info = self.osc.step(action_complex)
        obs = np.asarray(state).reshape(-1)  # now (3,)
        return obs, reward, terminated, False, info

    def render(self, mode='human'):
        # Optional: implement visualization
        pass

    def close(self):
        pass
