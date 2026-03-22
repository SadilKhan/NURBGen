class BsplineParameters:
    def __init__(self, poles, u_knots, v_knots, u_mults, v_mults,
                 u_degree, v_degree, u_periodic, v_periodic,
                 num_poles_u, num_poles_v, weights):
        self.poles = poles
        self.u_knots = u_knots
        self.v_knots = v_knots
        self.u_mults = u_mults
        self.v_mults = v_mults
        self.u_degree = u_degree
        self.v_degree = v_degree
        self.u_periodic = u_periodic
        self.v_periodic = v_periodic
        self.num_poles_u = num_poles_u
        self.num_poles_v = num_poles_v
        self.weights = weights

    def __repr__(self):
        return (
            f"BsplineParameters(poles={self.poles}, "
            f"u_knots={self.u_knots}, v_knots={self.v_knots}, "
            f"u_mults={self.u_mults}, v_mults={self.v_mults}, "
            f"u_degree={self.u_degree}, v_degree={self.v_degree}, "
            f"u_periodic={self.u_periodic}, v_periodic={self.v_periodic}, "
            f"num_poles_u={self.num_poles_u}, num_poles_v={self.num_poles_v}, "
            f"weights={self.weights})"
        )

    def to_json(self):
        data = {
            'poles': self.poles,
            'u_knots': self.u_knots,
            'v_knots': self.v_knots,
            'u_mults': self.u_mults,
            'v_mults': self.v_mults,
            'u_degree': self.u_degree,
            'v_degree': self.v_degree,
            'u_periodic': self.u_periodic,
            'v_periodic': self.v_periodic,
            # 'num_poles_u': self.num_poles_u,
            # 'num_poles_v': self.num_poles_v,
            'weights': self.weights
        }
        return data

    @staticmethod
    def from_json(data):
        if 'weights' not in data:
            data['weights'] = []
        return BsplineParameters(
            poles=data['poles'],
            u_knots=data['u_knots'],
            v_knots=data['v_knots'],
            u_mults=data['u_mults'],
            v_mults=data['v_mults'],
            u_degree=data['u_degree'],
            v_degree=data['v_degree'],
            u_periodic=data['u_periodic'],
            v_periodic=data['v_periodic'],
            num_poles_u= len(list(data['poles'])),
            num_poles_v=len(list(data['poles'][0])),
            weights=data['weights'] 
        )

