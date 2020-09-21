from scaler_base import *
from catmap.data import regular_expressions
import numpy as np

class ThermodynamicScaler(ScalerBase):
    """Scaler which uses temperature/pressure/potential as descriptors and 
    treates energetics as a constant"""
    def __init__(self,reaction_model=None):
        ScalerBase.__init__(self,reaction_model)

    def get_electronic_energies(self,descriptors):
        if len(self.surface_names) > 1:
            raise IndexError('Thermodynamic scaler works only with a \
                    single surface.')

        if self.adsorbate_interaction_model not in [None,'ideal']:
            raise NotImplementedError('Thermodynamic scaler modified. \
                    Adsorbate interaction not tested.')

        transition_state_energies = \
            [self.species_definitions[species]['formation_energy'][0]
             for species in self.transition_state_names]
        adsorbate_energies = \
            [self.species_definitions[species]['formation_energy'][0]
             for species in self.adsorbate_names] + [1]
        if any([val == -10.0 or val == None for val in transition_state_energies]):
            if self.transition_state_scaling_matrix is None:
                self.get_transition_state_scaling_matrix()
            transition_state_scaling_energies = \
                np.dot(self.transition_state_scaling_matrix, adsorbate_energies)
        energy_dict = {}
        for species in self.species_definitions:
            if species in self.adsorbate_names:
                energy_dict[species] = \
                    self.species_definitions[species]['formation_energy'][0]
            elif species in self.transition_state_names:
                if self.species_definitions[species]['formation_energy'][0] == None or \
                    self.species_definitions[species]['formation_energy'][0] == -10.0:
                    idx = self.transition_state_names.index(species)
                    energy_dict[species] = transition_state_scaling_energies[idx]
                else:
                    energy_dict[species] = self.species_definitions[species]['formation_energy'][0]
            elif species in self.gas_names+self.site_names:
                energy_dict[species] = self.species_definitions[species]['formation_energy']
        return energy_dict

    ### function adapted from GeneralizedLinearScaler
    def get_transition_state_scaling_matrix(self):
        """
            :TODO:
        """
        def state_scaling(TS,params,mode):
            coeffs = [0]*len(self.adsorbate_names)
            rxn_def = None
            for rxn in self.elementary_rxns:
                if len(rxn) == 3:
                    if TS in rxn[1]:
                        if rxn_def is None:
                            rxn_def = rxn
                        else:
                            rxn_def = rxn
                            print('Warning: ambiguous IS for '+TS+\
                                 '; Using'+self.print_rxn(rxn,mode='text'))
            if rxn_def is None:
                raise ValueError(TS+' does not appear in any reactions!')
            if mode == 'final_state':
                FS = rxn_def[-1]
                IS = []
            elif mode == 'initial_state':
                FS = rxn_def[0]
                IS = []
            elif mode == 'BEP':
                IS = rxn_def[0]
                FS = rxn_def[-1]
            else:
                raise ValueError('Invalid Mode')

            def get_energy_list(state,coeff_sign):
                energies = []
                for ads in state:
                    if ads in self.adsorbate_names:
                        idx = self.adsorbate_names.index(ads)
                        coeffs[idx] += coeff_sign
                    Ef = self.species_definitions[ads]['formation_energy']
                    if hasattr(Ef,'__iter__'):
                        energies.append(Ef)
                    else:
                        energies.append([0])
                return energies

            IS_energies = get_energy_list(IS,-1)
            FS_energies = get_energy_list(FS,+1)

            if params and len(params) == 2:
                m,b = [float(pi) for pi in params]
            else:
                raise ValueError('Transition state scaling not properly defined'\
                                  'for '+TS+'.')

            if mode == 'BEP':
                coeff_vals = []
                for k,ck in enumerate(coeffs):
                    ads = self.adsorbate_names[k]
                    if ads in IS:
                        coeff_vals.append((1.-m)*abs(ck))
                    elif ads in FS:
                        coeff_vals.append(m*abs(ck))
                    else:
                        coeff_vals.append(ck)
                offset = 0.
                for gas in self.gas_names:
                    if gas in IS:
                        offset += (1.-m)*self.species_definitions[gas]['formation_energy']
                    elif gas in FS:
                        offset += m*self.species_definitions[gas]['formation_energy']
                    else:
                        continue
                coeff_vals.append(b+offset)
            else:
                coeff_vals = [m*ci for ci in coeffs] + [b]

            return [m,b],coeff_vals

        def initial_state_scaling(TS,params):
            return state_scaling(TS,params,'initial_state')

        def final_state_scaling(TS,params):
            return state_scaling(TS,params,'final_state')

        def BEP_scaling(TS,params):
            return state_scaling(TS,params,'BEP')

        TS_scaling_functions = {
                'initial_state':initial_state_scaling,
                'final_state':final_state_scaling,
                'BEP':BEP_scaling,
                }

        TS_matrix = []
        TS_coeffs = []
        for TS in self.transition_state_names:
            if TS in self.scaling_constraint_dict:
                constring = self.scaling_constraint_dict[TS]
                if not isinstance(constring,basestring):
                    raise ValueError('Constraints must be strings: '\
                            +repr(constring))
                match_dict = self.match_regex(constring,
                    *regular_expressions[
                        'transition_state_scaling_constraint'])
                if match_dict is None:
                    raise ValueError('Invalid constraint: '+constring)
                mode = match_dict['mode']
                if match_dict['parameter_list']:
                    parameter_list = eval(match_dict['parameter_list'])
                else:
                    raise ValueError('Invalid constraint: '+constring)

                params = parameter_list
            else:
                raise KeyError('The '+TS+' must be defined '+\
                                'in scaling_constraint_dict')

            try:
                mb,coeffs=TS_scaling_functions[mode](TS,params)
                TS_matrix.append(coeffs)
                TS_coeffs.append(mb)
            except KeyError:
                raise NotImplementedError(
                        'Invalid transition-state scaling mode specified')

        TS_matrix = np.array(TS_matrix)
        self.transition_state_scaling_matrix = TS_matrix
        self.transition_state_scaling_coefficients = TS_coeffs
        return TS_matrix

    def get_thermodynamic_energies(self,descriptors,**kwargs):
        thermo_state = {}
        #synchronize all thermodynamic varibles
        for var,val in zip(self.descriptor_names,descriptors):
            thermo_state[var] = val
            setattr(self,var,val)
        if 'pressure' in self.descriptor_names:
            P = thermo_state['pressure']
        elif 'logPressure' in self.descriptor_names:
            P = 10**thermo_state['logPressure']
        else:
            P = 1

        if 'pressure' in self.descriptor_names or 'logPressure' in self.descriptor_names:
            if self.pressure_mode == 'static':
                #static pressure doesn't make sense if
                #pressure is a descriptor
                self.pressure_mode = 'concentration'

        self.pressure = P

        thermo_dict =  self.thermodynamics.get_thermodynamic_corrections(
                **kwargs)

        for key in self.site_names:
            if key not in thermo_dict:
                thermo_dict[key] = 0
        return thermo_dict

    def get_rxn_parameters(self,descriptors, *args, **kwargs):
        if self.adsorbate_interaction_model not in ['ideal',None]:
            params =  self.get_formation_energy_interaction_parameters(descriptors)
            return params
        else:
            params = self.get_formation_energy_parameters(descriptors)
            return params

    def get_formation_energy_parameters(self,descriptors):
        self.parameter_names = self.adsorbate_names + self.transition_state_names
        free_energy_dict = self.get_free_energies(descriptors)
        params =  [free_energy_dict[sp] for sp in self.adsorbate_names+self.transition_state_names]
        return params

    def get_formation_energy_interaction_parameters(self,descriptors):
        E_f = self.get_formation_energy_parameters(descriptors)
        if self.interaction_cross_term_names:
            param_names = self.adsorbate_names + self.interaction_cross_term_names
        else:
            param_names = self.adsorbate_names
        
        if not self.interaction_parameters:
            info = self.thermodynamics.adsorbate_interactions.get_interaction_info()
            params = [info[pi][0] for pi in param_names]
            params_valid = []
            for p,pname in zip(params,param_names):
                if p is not None:
                    params_valid.append(p)
                else:
                    raise ValueError('No interaction parameter specified for '+pname)
            self.interaction_parameters = params_valid

        epsilon = self.thermodynamics.adsorbate_interactions.params_to_matrix(E_f+self.interaction_parameters)
        epsilon = list(epsilon.ravel())
        return E_f + epsilon
