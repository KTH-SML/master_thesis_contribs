### ONE STEP VERSION

#! /usr/bin/env python3

import casadi as ca
import numpy as np
import os
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import rospy
import threading

from svea.helpers import load_param

class MPC_casadi:

    X0, XN = -4, 8
    Y0, YN = -4, 8

    def __init__(self, vehicle_name='', config_ns='~mpc'):
        """
        This is the release 0 of a general-purpose Nonlinear Model Predictive Controller (NMPC) 
        designed for the SVEA platform. It offers the following features:

        - Nonlinear dynamics with state representation [x, y, theta, v, steering] 
        and control inputs [steering_rate, acceleration].
        - Accepts a reference trajectory as input (see `compute_control` method) 
        and optimizes the predicted trajectory to minimize the deviation of 
        the predicted points from the next reference point.
        - Functions as both an optimal path planner and path tracker:
            - For path planning and tracking: Reference points should be spaced apart to give 
                the MPC the freedom to determine the optimal path.
            - For tracking only: Adjust parameters such as prediction horizon to 
                follow the reference trajectory closely.

        Limitations:
        - Neither static nor dynamic obstacles are considered 
        in this implementation.

        Initialize the MPC controller with the given parameters:

        :param L: Wheelbase of the vehicle (unit [m])
        :param N = 1: One step prediction
        :param dt: Sampling time
        :param min_steering: Minimum steering angle [rad]
        :param max_steering: Maximum steering angle [rad]
        :param min_acceleration: Minimum acceleration [m/s^2]
        :param max_acceleration: Maximum acceleration [m/s^2]
        :param min_velocity: Minimum velovity [m/s]
        :param max_velocity: Maximum velocity [m/s]
        :param Q1: State weight matrix (4x4)
        :param Q2: Control rate weight matrix (2x2)
        :param Q3: Control weight matrix (2x2)
        :param Qf: Final state weight matrix (4x4)
        :param Qv: Forward_speed_weight scalar
        """

        ## Core Parameters

        # The time step in which the optimization problem is divided (unit [s]).
        self.dt = ca.DM(load_param(f'{config_ns}/time_step'))
        
        ## Load weight matrices 
        # Note: we convert to dense matrix to allow symbolic operations.

        self.Q1_list = load_param(f'{config_ns}/state_weight_matrix')
        self.Q1 = ca.DM(np.array(self.Q1_list).reshape((4, 4)))

        self.Q2_list = load_param(f'{config_ns}/control_rate_weight_matrix')
        self.Q2 = ca.DM(np.array(self.Q2_list).reshape((2, 2)))

        self.Q3_list = load_param(f'{config_ns}/control_weight_matrix')
        self.Q3 = ca.DM(np.array(self.Q3_list).reshape((2, 2)))

        self.Qf_list = load_param(f'{config_ns}/final_state_weight_matrix')
        self.Qf = ca.DM(np.array(self.Qf_list).reshape((4, 4)))

        self.Qv_num  = load_param(f'{config_ns}/forward_speed_weight')
        self.Qv = ca.DM(self.Qv_num)

        self.Q_HJ = 10
        
        ## Model Parameters

        self.min_steering = np.radians(load_param(f'{config_ns}/steering_min'))
        self.max_steering = np.radians(load_param(f'{config_ns}/steering_max'))

        # self.min_steering_rate = np.radians(load_param(f'{config_ns}/steering_rate_min'))
        # self.max_steering_rate = np.radians(load_param(f'{config_ns}/steering_rate_max'))

        self.min_steering_rate = -np.pi/6
        self.max_steering_rate = np.pi/6

        self.min_velocity = load_param(f'{config_ns}/velocity_min')
        self.max_velocity = load_param(f'{config_ns}/velocity_max')

        self.min_acceleration = load_param(f'{config_ns}/acceleration_min')
        self.max_acceleration = load_param(f'{config_ns}/acceleration_max')

        self.min_bounds_data = np.array([ 0,           self.X0,           self.Y0,-np.pi, -np.pi/6, 0.0])
        self.max_bounds_data = np.array([20, self.X0 + self.XN, self.Y0 + self.YN, np.pi,  np.pi/6, 0.8])

        # Wheelbase of the vehicle (unit [m]).
        self.L = ca.DM(load_param(f'{config_ns}/wheelbase'))
        
        ## Setup CasADi
        self.opti = ca.Opti()

        ## Import value function and define grid
        self.set_value_function()
        dims = self.value_function.shape
        self.grids = [np.linspace(self.min_bounds_data[i], self.max_bounds_data[i], dims[i]) for i in range(6)]

        load_from_file = True
        if load_from_file: self.load_reachability_cost_function()      ## Load cost function from file
        else: self.define_reachability_cost_function()    ## Define cost function by interpoling value function

        ## Setup cost associated with value function
        self.define_state_and_control_variables()
        self.set_objective_function()
        self.set_state_constraints()
        self.set_control_input_constraints()
        self.set_solver_options()
        
        ## Publish points for reachability
        self.thread1 = threading.Thread(target=self.publish_red_dots_array, daemon = True)
        self.thread1.start()

        ## Publish points for reachability
        self.thread2 = threading.Thread(target=self.publish_value_func, daemon = True)
        # self.thread2.start()

        # for i in range(10):
        #     random_index = [
        #         np.random.random_integers(0, dims[i]-1)
        #         for i in range(len(dims))
        #     ]
        #     point = [self.grids[i][int(random_index[i])] for i in range(len(random_index))]

        #     print("Index:", random_index)
        #     print("Point:", point)
        #     print("Value function at random index:", self.value_function[random_index[0],random_index[1],random_index[2],random_index[3],random_index[4],random_index[5]])
        #     print("Reachability cost function at random point:", self.reachability_cost_function(ca.vertcat(point)), "\n")

        ## Publish points for warmstart
        self.warmstart = []
        self.thread3 = threading.Thread(target=self.publish_warmstart, daemon = True)
        self.thread3.start()

    def compute_control(self, state, reference_trajectory):
        """
        Compute the control actions (steering, acceleration) using MPC.
        
        :param state: Current state of the vehicle [x, y, theta, v, delta]
        :param reference_trajectory: Reference trajectory [4, N+1]
        :return: steering, velocity
        """
        # Bound the initial state to respect constraints
        bounded_state = self.bound_initial_state(state)

        # Enlarge the reference trajectory with a fictitious steering value for feasibility.
        # Add one rows of zero to the reference_trajectory matrix.
        reference_trajectory = ca.vertcat(reference_trajectory,ca.DM.zeros(1, reference_trajectory.shape[1]))

        # Set current state and reference trajectory for the active part of the horizon
        # print(f"reference_trajectory shape: {reference_trajectory.shape}")
        # print(f"self.current_horizon: {self.current_horizon}")
        # print(f"Attempting slice: reference_trajectory[:, :{self.current_horizon+1}]")

        now = rospy.get_rostime()
        self.warmstart = self.compute_warmstart([0,state[0],state[1],state[2],state[3],state[4]])
        print("Warmstart in ",(rospy.get_rostime() - now).to_sec(),"sec")

        self.opti.set_value(self.x_init, bounded_state)
        self.opti.set_value(self.x_ref[:, :self.current_horizon+1], reference_trajectory[:, :self.current_horizon+1])

        # self.opti.set_initial(self.x, self.warmstart[1:6])

        # Extract control actions (acceleration and steering rate)
        try:
            # Solve the optimization problem
            self.sol = self.opti.solve()
            acceleration = self.sol.value(self.u[0, 0])
            steering_rate = self.sol.value(self.u[1, 0])
        except:
            acceleration = 0
            steering_rate = 0
            print("No solve")

        return steering_rate, acceleration
    
    def get_optimal_control(self, all=True):
        """
        This method returns the optimal control computed by the mpc.
        :param all: Flag to indicate if all of the optimal control for the 
            whole prediction should	be returned or only the first step.
        :type all: bool
        :return: The optimal control computed by the mpc controller.
        :rtype: NumPy array
        """
        if self.sol is not None:
            u_opt = np.array(self.sol.value(self.u))
            if all:
                return u_opt
            else:
                return u_opt[:,0]
        else:
            return None
        
    def get_optimal_states(self):
        """
        This method returns the optimal states computed by the mpc controller
        for the whole prediction horizon.
        :return: The optimal states for the whole prediction horizon.
        :rtype: NumPy array
        """
        if self.sol is not None:
            return np.array(self.sol.value(self.x))   # Note: to get only optimized one: self.sol.value(self.x[:, :self.current_horizon])
        else: 
            return None

    def define_state_and_control_variables(self):
        # Define state and control variables
        self.x = self.opti.variable(5, 2)  # state = [x, y, theta, v, steering]
        self.u = self.opti.variable(2, 1)      # input = [steering_rate, acceleration]

        self.x_init = self.opti.parameter(5)             # Initial state
        self.x_ref = self.opti.parameter(5, 2)  # Reference trajectory

    def set_objective_function(self):
        # Define the objective function: 
        # J = (x[k]-x_ref[k])^T Q1 (x[k]-x_ref[k]) + (u[k+1] - u[k])^T Q2 (u[k+1] - u[k]) + u[k]^T Q3 u[k] + Qv max(0,-x[3])^2
        self.objective = 0

        # State error term (ignore delta in reference trajectory)
        state_error = self.compute_state_error(self.x[:, 1], self.x_ref[:, 1])

        # Control input rate of change term (x[1] - x[0])
        change_of_rate_cost = self.x[:, 1] - self.x[:, 0]

        # Penalize for negative velocity (soft constraint)
        velocity_penalty = ca.fmax(0, -self.x[3, 1])  # Penalize if v < 0

        # Accumulate the terms into the objective
        self.objective += (ca.mtimes([state_error.T, self.Q1, state_error]) 
                            # + ca.mtimes([change_of_rate_cost.T, self.Q2, change_of_rate_cost])
                            + ca.mtimes([self.u[:, 0].T, self.Q3, self.u[:, 0]])
                            + ca.mtimes([velocity_penalty.T, self.Qv, velocity_penalty]))
            
        # Final state cost
        # final_state_error = self.compute_state_error(self.x[:, self.current_horizon], self.x_ref[:, self.current_horizon])
        # self.objective += ca.mtimes([final_state_error.T, self.Qf, final_state_error])

        # Specify type of optimization problem
        self.opti.minimize(self.objective)

    def set_state_constraints(self):
        # Initial state constraint
        self.opti.subject_to(self.x[:, 0] == self.x_init)

        # Vehicle dynamics constraints - Simple kinematic bycicle model
        x_next = self.x[0, 0] + self.dt * self.x[3, 0] * ca.cos(self.x[2, 0])                   # x_k+1 = x_k + dt * v_k * cos(theta_k)
        y_next = self.x[1, 0] + self.dt * self.x[3, 0] * ca.sin(self.x[2, 0])                   # y_k+1 = y_k + dt * v_k * sin(theta_k)
        theta_next = self.x[2, 0] + self.dt * (self.x[3, 0] / self.L) * ca.tan(self.x[4, 0])    # theta_k+1 = theta_k + dt * v_k * tan(delta_k) / L
        v_next = self.x[3, 0] + self.dt * self.u[0, 0]                                          # v_k+1 = v_k + dt * a_k
        delta_next = self.x[4, 0] + self.dt * self.u[1, 0]                                      # delta_k+1 = delta_k + dt * steering_rate_k

        self.opti.subject_to(self.x[0, 1] == x_next)
        self.opti.subject_to(self.x[1, 1] == y_next)
        self.opti.subject_to(self.x[2, 1] == theta_next)
        self.opti.subject_to(self.x[3, 1] == v_next)
        self.opti.subject_to(self.x[4, 1] == delta_next)

        # self.opti.subject_to(
        #     self.reachability_cost_function(
        #         ca.vertcat(0, self.x[0, k+1], self.x[1, k+1], self.x[2, k+1], self.x[3, k+1], self.x[4, k+1])
        #     ) < 0
        # )

        # Position constraints
        self.opti.subject_to(self.x[0, 0] <= self.XN)
        self.opti.subject_to(self.x[0, 0] >= self.X0)
        self.opti.subject_to(self.x[1, 0] <= self.YN)
        self.opti.subject_to(self.x[1, 0] >= self.Y0)

        # Velocity constraints
        self.opti.subject_to(self.x[3, 0] <= self.max_velocity)
        self.opti.subject_to(self.x[3, 0] >= self.min_velocity)

        # Steering angle constraints
        self.opti.subject_to(self.x[4, 0] <= self.max_steering)
        self.opti.subject_to(self.x[4, 0] >= self.min_steering)
    
    def set_control_input_constraints(self):
        # Input constraints (acceleration, steering rate)
        self.opti.subject_to(self.min_acceleration <= self.u[0, 0])
        self.opti.subject_to(self.u[0, 0] <= self.max_acceleration)

        # Steering rate constraint
        self.opti.subject_to(self.min_steering_rate <= self.u[1, 0])
        self.opti.subject_to(self.u[1, 0] <= self.max_steering_rate)

    def set_solver_options(self):
        # Set solver options
        opts = {"ipopt.print_level": 0, "print_time": 0}
        self.opti.solver("ipopt", opts)

    def load_reachability_cost_function(self):
        """
        Loads reachability function directly from saved file
        """
        self.reachability_cost_function = ca.Function.load("reachability_cost_function.casadi")

    def set_value_function(self):
        # Import value function from file
        file_path = os.path.join(os.path.dirname(__file__), "out1.npy")
        self.value_function = np.load(file_path)
        print("Loaded value function of type",type(self.value_function),"of shape",self.value_function.shape)

    def define_reachability_cost_function(self):
        """
        This converts the value-function-grid into a continuous function using interpolants.
        The function is saved so that it can be loaded in efficiently if no changes are made\n
        """
        k = 1
        t = 0
        cost = 0
        dx, dy = self.XN/len(self.value_function[t]), self.YN/len(self.value_function[t][0])
        p = 10/dx

        X0, Y0 = self.X0, self.Y0
        XN, YN = self.XN, self.YN

        x_var = self.opti.variable()
        y_var = self.opti.variable()
        t_var = self.opti.variable()

        print("Loading reachability function")

        now = rospy.get_rostime()

        # Create CasADi interpolation function
        self.reachability_cost_function = ca.interpolant("reachability_cost_function", "linear", self.grids, self.value_function.ravel(order="F"))  # Column-major flattening

        save_path = os.path.join(os.getcwd(), "reachability_cost_function.casadi")
        self.reachability_cost_function.save(save_path)

        time = (rospy.get_rostime() - now).to_sec()

        print(f"Function saved at: {save_path}, load time: {time}")

    def publish_red_dots_array(self):
        marker_pub = rospy.Publisher("/visualization_marker_array", MarkerArray, queue_size=10)

        rate = rospy.Rate(1)  # Publish at 1 Hz

        # Define multiple positions for the red dots
        t = 0
        positions = []
        dx, dy = self.XN/len(self.value_function[t]), self.YN/len(self.value_function[t][0])

        for i in range(len(self.value_function[t])):  # Iterate over row indices
            for j in range(len(self.value_function[t][i])):  # Iterate over column indices
                # Check if the minimum value in self.value_function[t][i][j] is less than 0

                if np.min(self.value_function[t][i][j]) >= 0:
                    positions.append((self.X0+i*dx,self.Y0+j*dy,0))
        
        while not rospy.is_shutdown():
            marker_array = MarkerArray()  # Create an array of markers

            for i, (x, y, z) in enumerate(positions):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "red_dots"
                marker.id = i  # Each marker must have a unique ID
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD

                # Set position
                marker.pose.position = Point(x, y, z)
                marker.pose.orientation.w = 1.0

                # Set scale (size of dots)
                s = (dx+dy)/2
                marker.scale.x = s
                marker.scale.y = s
                marker.scale.z = s

                # Set color (red, full opacity)
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                marker.color.a = 1.0

                marker.lifetime = rospy.Duration()  # Keep dots persistent

                marker_array.markers.append(marker)

            # Publish the entire marker array
            marker_pub.publish(marker_array)
            
            rate.sleep()
    
    def publish_value_func(self):
        marker_pub = rospy.Publisher("/visualization_cost_func", MarkerArray, queue_size=10)

        rate = rospy.Rate(5)  # Publish at 5 Hz

        # Define multiple positions for the red dots
        positions_list = []
        grid_size = 50
        V_shape = self.value_function.shape

        dx, dy = self.XN/V_shape[1], self.YN/V_shape[2]

        eval_points = [[h,v,yr] for h in range(V_shape[3]) for v in range(V_shape[4]) for yr in range(V_shape[5])]
        # print(eval_points)

        print("Solving reachability min")

        # Now you can evaluate it at a specific (t, x, y)
        for t in range(V_shape[0]):
            positions = []
            for i in range(V_shape[1]):  # Iterate over row indices
                for j in range(V_shape[2]):  # Iterate over column indices
                    positions.append((self.X0+i*dx,self.Y0+j*dy,np.min(self.value_function[t][i][j])))
            positions_list.append(positions)

        # print(positions_list)
        # for time in range(len(self.value_function)):
        #     for i in range(len(self.value_function[time])):  # Iterate over row indices
        #         for j in range(len(self.value_function[time][i])):  # Iterate over column indices
        #             # Check if the minimum value in self.value_function[t][i][j] is less than 0
        #             if np.min(self.value_function[time][i][j]) < 0:
        #                 positions.append((X0+i*dx,Y0+j*dy,time/5))

        # positions = [(x * 0.5, x * 0.5, 0.5) for x in range(10)]  # Generates a diagonal line of dots
        print("Creating markers")

        marker_array_array = []

        for positions in positions_list:

            marker_array = MarkerArray()  # Create an array of markers

            for i, (x, y, z) in enumerate(positions):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "red_dots"
                marker.id = i  # Each marker must have a unique ID
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD

                # Set position
                marker.pose.position = Point(x, y, z)
                marker.pose.orientation.w = 1.0

                # Set scale (size of dots)
                s = (dx+dy)/2
                marker.scale.x = s
                marker.scale.y = s
                marker.scale.z = s

                # Set color (red, full opacity)
                if z >= 0.0:
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                else:
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                marker.color.b = 0.0
                marker.color.a = 1.0

                marker.lifetime = rospy.Duration()  # Keep dots persistent

                marker_array.markers.append(marker)

            marker_array_array.append(marker_array)

        # print(marker_array_array)

        print("Solved reachability-min")
        t = 0
        
        while not rospy.is_shutdown():
            if t == len(positions_list):
                t = 0
            # Publish the entire marker array
            marker_pub.publish(marker_array_array[t])
            
            rate.sleep()
            t += 1
    
    def publish_warmstart(self):
        marker_pub = rospy.Publisher("/visualization_warmstart", MarkerArray, queue_size=10)

        rate = rospy.Rate(1)  # Publish at 1 Hz

        # Define multiple positions for the red dots
        t = 0
        positions = []

        # print(warmstart)
        for pos in self.warmstart:
            positions.append((pos[1],pos[2],pos[0]/4*0))
        
        while not rospy.is_shutdown():
            marker_array = MarkerArray()  # Create an array of markers

            for i, (x, y, z) in enumerate(positions):
                marker = Marker()
                marker.header.frame_id = "map"
                marker.header.stamp = rospy.Time.now()
                marker.ns = "blue_dots"
                marker.id = i  # Each marker must have a unique ID
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD

                # Set position
                marker.pose.position = Point(x, y, z)
                marker.pose.orientation.w = 1.0

                # Set scale (size of dots)
                s = 0.1
                marker.scale.x = s
                marker.scale.y = s
                marker.scale.z = s

                # Set color (red, full opacity)
                marker.color.r = 0.0
                marker.color.g = 0.0
                marker.color.b = 1.0
                marker.color.a = 1.0

                marker.lifetime = rospy.Duration()  # Keep dots persistent

                marker_array.markers.append(marker)

            # Publish the entire marker array
            marker_pub.publish(marker_array)
            
            rate.sleep()

    def bound_initial_state(self,state):
        """
        This method checks if the initial state provided to the mpc, which could come from the localization stack,
        is within the allowed bounds. If not, it clamps it to make the optimization problem feasible.
        """
        # Ensure the velocity is within the specified bounds
        clamped_velocity = max(self.min_velocity, min(state[3], self.max_velocity))
        
        # Create a new state with the clamped velocity
        bounded_state = state.copy()
        bounded_state[3] = clamped_velocity
        
        return bounded_state
    
    def compute_state_error(self, x, x_ref):
        """
        Computes the state error between the current state and the reference state.
        Adjusts the yaw error to account for angle wrapping in the range [-π, π].
        """
        state_error = x[0:4] - x_ref[0:4]
        yaw_diff = x[2] - x_ref[2]
        state_error[2] = ca.atan2(ca.sin(yaw_diff), ca.cos(yaw_diff))  # Minimum yaw error 
        return state_error

    def set_new_weight_matrix(self, matrix_name, new_value):
        """
        Dynamically adjust one of the weight matrices.
        Check if the matrix exists as an attribute and if so, update it.
        """
        if hasattr(self, matrix_name):
            try:
                # Overwrite with the new dense matrix
                setattr(self, matrix_name, ca.DM(new_value))
                # Redefine objective function with updated matrix.
                self.set_objective_function()
            except Exception as e:
                print(f"Failed to update {matrix_name}: {e}")
        else:
            print(f"Matrix {matrix_name} does not exist in MPC class.")

    def set_new_prediction_horizon(self, new_horizon):
        """
        Dynamically adjust the active horizon for the optimization problem.
        When the horizon is reduced, you simply freeze unused variables and update the reference trajectory 
        and state for the active part of the horizon. The solver will then only optimize over the active steps.
        """
        self.current_horizon = new_horizon
        
        # Redefine objective function with new horizon.
        self.set_objective_function()

    def reset_parameters(self):
        """
        Reset the core parameters and weight matrices of the MPC instance to their initial values.
        Useful for restoring values to their original state after runtime modifications.
        """
        self.current_horizon = 1  # Reset to max horizon

        # Reset weight matrices
        self.Q1 = ca.DM(np.array(self.Q1_list).reshape((4, 4)))
        self.Q2 = ca.DM(np.array(self.Q2_list).reshape((2, 2)))
        self.Q3 = ca.DM(np.array(self.Q3_list).reshape((2, 2)))
        self.Qf = ca.DM(np.array(self.Qf_list).reshape((4, 4)))
        self.Qv = ca.DM(self.Qv_num)
        # Reset objective function with initial values.
        self.set_objective_function()

    def update_step(self, state, control, min_bounds, max_bounds):
        L = 0.8  # Wheelbase of the vehicle
        tau = 0.2  # Time constant

        x = state[1]        # x-coordinate
        y = state[2]        # y-coordinate
        theta = state[3]    # Yaw/Heading
        omega = state[4]    # Change rate of yaw
        v = state[5]        # Velocity
        
        delta = control[0]  # Steering angle
        a = control[1]      # Acceleration

        # Compute the state derivatives
        dt = 0.2
        dx = v * ca.cos(theta)
        dy = v * ca.sin(theta)
        dtheta = omega
        domega = v/L*np.tan(delta)-omega/tau
        dv = a

        dstate = [0,dx,dy,dtheta,domega,dv]
        new_state = [float(state[i] + dt*dstate[i]) for i in range(6)]
        return [min(max_bounds[i], max(min_bounds[i], new_state[i])) for i in range(6)]

    def compute_warmstart(self, start_point):
        path = [start_point]
        nr_of_samples = 20
        t_max = 20

        min_bounds_state = [ 0,         self.X0,         self.Y0, -np.pi, -np.pi/6, 0.0]
        max_bounds_state = [20, self.X0+self.XN, self.Y0+self.YN,  np.pi,  np.pi/6, 0.8]

        min_bounds_input = [-5*np.pi/4, -0.4]
        max_bounds_input = [ 5*np.pi/4,  0.4]

        point = start_point
        reached_end = False

        dt = 0.2

        for _ in range(10):
            while self.reachability_cost_function(ca.vertcat(point)) < 0:
                point[0] += dt

                if point[0] >= t_max:
                    reached_end = True
                    break

            if reached_end:
                break

            values = []
            random_points = []
            for i in range(nr_of_samples):
                control = np.random.uniform(min_bounds_input, max_bounds_input)
                new_point = self.update_step(point, control, min_bounds_state, max_bounds_state)
                random_points.append(new_point)
                
                size = 0.5
                value = -np.inf
                for dx in [-size, 0, size]:
                    for dy in [-size, 0, size]:
                        new_value = self.reachability_cost_function(ca.vertcat(new_point[0],new_point[1]+dx,new_point[2]+dy,new_point[3],new_point[4],new_point[5]))
                        if new_value > value:
                            value = new_value

                values.append(value)
                
            min_index = values.index(min(values))
            best_point = random_points[min_index]

            path.append(best_point)
            point = best_point

        return path
    
    def compute_warmstart_OLD(self):
        STATE_INIT = [self.X0+7.5*self.XN/8, self.Y0+1.5*self.YN/8, np.pi, 0, 0]
        start_point = [0] + STATE_INIT
        path = [start_point]
        nr_of_samples = 100
        t_max = 20

        min_bounds_state = [ 0,         self.X0,         self.Y0, -np.pi, -np.pi/6, 0.0]
        max_bounds_state = [20, self.X0+self.XN, self.Y0+self.YN,  np.pi,  np.pi/6, 0.8]

        min_bounds_input = [-5*np.pi/4, -0.4]
        max_bounds_input = [ 5*np.pi/4,  0.4]

        point = start_point
        reached_end = False

        dt = 0.2

        for _ in range(100):
            while self.value_function[self.coord_to_index(point)] < 0:
                point[0] += dt

                if point[0] >= t_max:
                    reached_end = True
                    break

            if reached_end:
                break

            random_points = []
            for i in range(nr_of_samples):
                control = np.random.uniform(min_bounds_input, max_bounds_input)
                
                new_point = self.update_step(point, control, min_bounds_state, max_bounds_state)
                random_points.append(new_point)

            random_indexes = []
            for p in random_points:
                random_indexes.append(self.coord_to_index(p))
            # print(random_indexes)

            values = [self.value_function[ri] for ri in random_indexes]

            min_index = values.index(min(values))
            best_point = random_points[min_index]

            path.append(best_point)
            point = best_point

        return path
    
    def coord_to_index(self, coords):
        indices = []
        out1_shape = self.value_function.shape
        for i in range(6):
            norm_value = (coords[i] - self.min_bounds_data[i]) / (self.max_bounds_data[i] - self.min_bounds_data[i])
            index = round(norm_value * (out1_shape[i] - 1))
            index = max(0, min(index, out1_shape[i] - 1))
            indices.append(int(index))

        return tuple(indices)