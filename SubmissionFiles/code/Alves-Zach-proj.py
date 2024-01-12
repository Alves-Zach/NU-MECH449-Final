import numpy as np
import modern_robotics as mr
import csv
import matplotlib.pyplot as plt
np.set_printoptions(linewidth=180)
np.set_printoptions(suppress=True)

# Each line in the CSV should look like
# chassis phi, chassis x, chassis y, J1, J2, J3, J4, J5, W1, W2, W3, W4, gripper state

class Arm():
    def __init__(self):
        # Defining default parameters when robot is home
        # Thetalist = (0, 0, 0, 0, 0, 0) at home
        self.Mhome = np.array([[1, 0, 0, 0.033],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0.6546],
                          [0, 0, 0, 1]])
        
        self.B1 = np.array([0, 0, 1, 0, 0.033, 0])
        self.B2 = np.array([0, -1, 0, -0.5076, 0, 0])
        self.B3 = np.array([0, -1, 0, -0.3526, 0, 0])
        self.B4 = np.array([0, -1, 0, -0.2176, 0, 0])
        self.B5 = np.array([0, 0, 1, 0, 0, 0])
        self.Blist = np.stack([self.B1, self.B2, self.B3, self.B4, self.B5], axis=1)

class Chassis():
    def __init__(self):
        self.r = 0.0475
        self.l = 0.47/2
        self.w = 0.3/2
        
        self.Tb0 = np.array([[1, 0, 0, 0.1662],
                             [0, 1, 0,      0],
                             [0, 0, 1, 0.0026],
                             [0, 0, 0,      1]])
        
        self.F = np.array([[-1/(self.l + self.w), 1/(self.l + self.w), 1/(self.l + self.w), -1/(self.l + self.w)],
                           [1, 1, 1, 1],
                           [-1, 1, -1, 1]])
        self.F = (self.r/4) * self.F

        self.F6 = np.append(np.zeros((2, 4)), self.F, axis=0)
        self.F6 = np.append(self.F6, np.zeros((1, 4)), axis=0)
       
def NextState(curConfig, curSpeeds, dt, F, wLimit=0.0):
    """
    - Args:
        - curConfig: A 12-vector representing the current configuration of the robot 
            (3 variables for the chassis configuration, 5 variables for the arm configuration, 
            and 4 variables for the wheel angles)
        - curSpeeds: A 9-vector of controls indicating the wheel speeds u (4 variables)
            and the arm joint speeds θ (5 variables)
        - dt: The time step
        - wLimit: The velocity limit for both angular velocity for the arms and the wheels
            Ex: wLimit = 12.3 -> Upper limit is 12.3 Lower limit is -12.3
        
    - Returns:
        - newChassisConfig [1x3]: odometry stuff
        - newJointAngles [1x5]: curConfig(jointAngles) + wLimit * dt
        - newWheelAngles [1x4]: curConfig(wheelAngles) + wLimit * dt
    """
    
    # Changing the w values to their limit if they're over the limits
    if wLimit != 0.0:
        for i in range(len(curSpeeds)):
            if curSpeeds[i] > wLimit:
                curSpeeds[i] = wLimit
            if curSpeeds[i] < -wLimit:
                curSpeeds[i] = -wLimit

    newJointAngles = curConfig[3:8] + curSpeeds[4:9] * dt
    newWheelAngles = curConfig[8:12] + curSpeeds[0:4] * dt

    # Odometry things
    
    # pinv(H) @ u = qdot
    # newOdom = curOdom + qdot * dt
    
    # dtheta = np.array(newWheelAngles - curConfig[8:12])
    dtheta = curSpeeds[0:4] * dt
    Vb = F @ dtheta
    
    rotMat = np.array([[1,                   0,                      0],
                       [0, np.cos(curConfig[0]), -np.sin(curConfig[0])],
                       [0, np.sin(curConfig[0]),  np.cos(curConfig[0])]])
    
    # Checking if wz == 0
    if Vb[0] == 0.0:
        # Add to newOdom without converting
        newOdom = curConfig[0:3] + rotMat @ np.array([0, Vb[1], Vb[2]])
    else:
        newOdom = curConfig[0:3] + rotMat @ np.array([Vb[0], 
                                                     (Vb[1] * np.sin(Vb[0]) + (Vb[2] * (np.cos(Vb[0]) - 1))) / Vb[0],
                                                     (Vb[2] * np.sin(Vb[0]) + (Vb[1] * (1 - np.cos(Vb[0])))) / Vb[0]])

    return newOdom, newJointAngles, newWheelAngles
    
def TrajectoryGenerator(Tse_init, Tsc_init, Tsc_final, Tce_grasp, Tce_standoff, k):
    """Generates the reference trajectories for the end effector frame {e}
    
    1. A trajectory to move the gripper from its initial configuration to a "standoff" configuration a few cm above the block.
    2. A trajectory to move the gripper down to the grasp position.
    3. Closing of the gripper.
    4. A trajectory to move the gripper back up to the "standoff" configuration.
    5. A trajectory to move the gripper to a "standoff" configuration above the final configuration.
    6. A trajectory to move the gripper to the final configuration of the object.
    7. Opening of the gripper.
    8. A trajectory to move the gripper back to the "standoff" configuration.
    
    Transformation matracies are input into the csv as such
    r11, r12, r13, r21, r22, r23, r31, r32, r33, px, py, pz, gripper state
    The array above would corrispond to the transform
    [[r11, r12, r13, px],
     [r21, r22, r23, py],
     [r31, r32, r33, pz],
     [  0,   0,   0,  1]]

    - Args:
        - Tse_init (np.array(4, 4)): The initial configuration of the EE relative to the initial chassis base
        - Tsc_init (np.array(4, 4)): The cube's initial configuration relative to the initial chassis base
        - Tsc_final (np.array(4, 4)): The desired cube final configuration relative to the initial chassis base
        - Tce_grasp (np.array(4, 4)): The EE configuration relative to the cube when grasping the cube
        - Tce_standoff (np.array(4, 4)): The EE configuration relative to the cube JUST BEFORE grasping the cube
        - k (int): Number of reference configurations per 0.01 seconds
        
    - Returns:
        - fullTraj (np.array(N, 13)): The full trajectory of the arm that will bring it through all steps
                                    where N is the number of lines it takes to get through all steps
    """
    print("Generating arm trajectory")
    
    # The list of transforms I will need to use
    # Assuming that the chassis will move so it's 0.5m from the cube and facing the cube
    Tsc_prime = Tsc_init
    Tsc_finalprime = Tsc_final @ Tce_grasp
    Tse_standoff = Tsc_prime @ Tce_standoff
    Tse_grasp = Tsc_prime @ Tce_grasp
    Tse_finalstandoff = Tsc_final @ Tce_standoff
    
    #####################################
    # GENERATING THE 1st TRAJECTORY
    #####################################
    # Creating the transformation matracies for Traj1
    Traj1T = mr.CartesianTrajectory(Tse_init, Tse_standoff,
                               4, (4*k)/0.01, 3)
    
    # Converting Traj1 to the array otuput Copellia can read
    Traj1 = []
    for i in Traj1T:
        Traj1.append([i[0][0], i[0][1], i[0][2],
                      i[1][0], i[1][1], i[1][2],
                      i[2][0], i[2][1], i[2][2], 
                      i[0][3], i[1][3], i[2][3], 0])
        
    #####################################
    # GENERATING THE 2nd TRAJECTORY
    #####################################
    # Creating the transformation matracies for Traj2
    Traj2T = mr.ScrewTrajectory(Tse_standoff, Tse_grasp, 4, (4*k)/0.01, 3)
    
    # Converting Traj2 to the array otuput Copellia can read
    Traj2 = []
    for i in Traj2T:
        Traj2.append([i[0][0], i[0][1], i[0][2],
                      i[1][0], i[1][1], i[1][2],
                      i[2][0], i[2][1], i[2][2], 
                      i[0][3], i[1][3], i[2][3], 0])
        
    #####################################
    # GENERATING THE 3rd TRAJECTORY
    #####################################
    # Traj3 is closing the gripper which takes about 0.625 seconds
    # Making this section take 0.63 seconds (63 lines in CSV) to give enough time to gripper
    Traj3 = np.zeros([63, 13])
    for i in range(63):
        Traj3[i] = np.copy(Traj2[len(Traj2) - 1])
        Traj3[i][12] = 1
    
    #####################################
    # GENERATING THE 4th TRAJECTORY
    #####################################
    # Creating the transformation matracies for Traj4
    Traj4T = mr.ScrewTrajectory(Tse_grasp, Tse_standoff, 4, (4*k)/0.01, 3)
    
    # Converting Traj2 to the array otuput Copellia can read
    Traj4 = []
    for i in Traj4T:
        Traj4.append([i[0][0], i[0][1], i[0][2],
                      i[1][0], i[1][1], i[1][2],
                      i[2][0], i[2][1], i[2][2], 
                      i[0][3], i[1][3], i[2][3], 1])
    
    #####################################
    # GENERATING THE 5th TRAJECTORY
    #####################################
    # Creating the transformation matracies for Traj4
    Traj5T = mr.ScrewTrajectory(Tse_standoff, Tse_finalstandoff, 6, (6*k)/0.01, 3)
    
    # Converting Traj4 to the array otuput Copellia can read
    Traj5 = []
    for i in Traj5T:
        Traj5.append([i[0][0], i[0][1], i[0][2],
                      i[1][0], i[1][1], i[1][2],
                      i[2][0], i[2][1], i[2][2], 
                      i[0][3], i[1][3], i[2][3], 1])
        
    #####################################
    # GENERATING THE 6th TRAJECTORY
    #####################################
    # Creating the transformation matracies for Traj4
    Traj6T = mr.ScrewTrajectory(Tse_finalstandoff, Tsc_finalprime, 6, (6*k)/0.01, 3)
    
    # Converting Traj4 to the array otuput Copellia can read
    Traj6 = []
    for i in Traj6T:
        Traj6.append([i[0][0], i[0][1], i[0][2],
                      i[1][0], i[1][1], i[1][2],
                      i[2][0], i[2][1], i[2][2], 
                      i[0][3], i[1][3], i[2][3], 1])
        
    #####################################
    # GENERATING THE 7th TRAJECTORY
    #####################################
    # Traj7 is closing the gripper which takes about 0.625 seconds
    # Making this section take 0.63 seconds (63 lines in CSV) to give enough time to gripper
    Traj7 = np.zeros([63, 13])
    for i in range(63):
        Traj7[i] = np.copy(Traj6[len(Traj6) - 1])
        Traj7[i][12] = 0
    
    #####################################
    # GENERATING THE 8th TRAJECTORY
    #####################################
    # Creating the transformation matracies for Traj4
    Traj8T = mr.ScrewTrajectory(Tsc_finalprime, Tse_finalstandoff, 3, (3*k)/0.01, 3)
    
    # Converting Traj4 to the array otuput Copellia can read
    Traj8 = []
    for i in Traj8T:
        Traj8.append([i[0][0], i[0][1], i[0][2],
                      i[1][0], i[1][1], i[1][2],
                      i[2][0], i[2][1], i[2][2], 
                      i[0][3], i[1][3], i[2][3], 0])
        
    ############
    # COMBINING ALL TRAJ# 
    fullTraj = []
    fullTraj = np.concatenate((Traj1, Traj2, Traj3, Traj4,
                               Traj5, Traj6, Traj7, Traj8))
    return fullTraj
    
def FeedbackControl(curConfig, Tse, Tsed_cur, Tsed_next, Kp, Ki,
                    dt, intErrin, arm=Arm, chassis=Chassis):
    """Generates the feedforward and feedback controll for the arm

    - Args:
        - curConfig:  A 12-vector representing the current configuration of the robot 
            (3 variables for the chassis configuration, 5 variables for the arm configuration, 
            and 4 variables for the wheel angles)
        - Tse (np.array(4, 4)): The actual end effector configuration
        - Tsed_cur (np.array(4, 4)): The desired end effector configuration at this time stamp
        - Tsed_next (np.array(4, 4)): The desired end effector configuration at the NEXT time stamp
        - Kp (np.array(6)): The proportional gain
        - Ki (np.array(6)): The integral gain
        - dt (float): The time step size
        
    - Returns:
        - V (np.array(4)): The commanded end-effector twist
        - intErrorOut (int): The calculated error including the previous error
        - velCmds (np.array(6, 1)): The commanded joint velocities
    """
    
    # The error in the current time step
    Terr = mr.MatrixLog6(mr.TransInv(Tse) @ Tsed_cur)
    TerrVec = mr.se3ToVec(Terr)
    
    # The twist Vd from the current transform to the desired transform
    Vd = (1/dt) * mr.MatrixLog6(mr.TransInv(Tsed_cur) @ Tsed_next)
    VdVec = mr.se3ToVec(Vd)
    
    # Creating the terms of the controller
    Term1 = mr.Adjoint(mr.TransInv(Tse) @ Tsed_cur) @ VdVec
    
    Term2 = Kp @ TerrVec
    
    # intErrout = intErrin + (Ki @ (TerrVec * dt))
    intErrout = (intErrin + TerrVec)
    
    Vt = Term1 + Term2 + (Ki @ intErrout * dt)
    
    # Calculating the jacobian
    T0ecur = mr.FKinBody(arm.Mhome, arm.Blist, curConfig[3:8])
    Jbase = mr.Adjoint(mr.TransInv(T0ecur) @ mr.TransInv(chassis.Tb0)) @ chassis.F6
    
    Jarm = mr.JacobianBody(arm.Blist, curConfig[3:8])
    
    Je = np.hstack((Jbase, Jarm))
    
    velCmds = np.linalg.pinv(Je) @ Vt
    
    # Print values to test
    # print(f"Vd: {VdVec}")
    # print(f"AdVd: {Term1}")
    # print(f"V: {Vt}")
    # print(f"Xerr: {TerrVec}")
    # print(f"Xerr * dt: {TerrVec * dt}")
    # print(f"Je: {Je}")
    # print(f"(u, theta): {velCmds}")
    
    return intErrout, velCmds, TerrVec
    
def testNextState1():
    # Starting config
    chassisConfig = np.array([0, 0, 0])
    armConfig = np.array([0, 0, 0, 0, 0])
    wheelConfig = np.array([0, 0, 0, 0])
    curConfig = np.concatenate([chassisConfig, armConfig, wheelConfig])
    
    wheelSpeeds = np.array([10, 10, 10, 10])
    armSpeeds = np.array([0, 0, 0, 0, 0])
    curSpeeds = np.concatenate([wheelSpeeds, armSpeeds])
    dt = 0.01
    wLimit = 10.0 # Setting just above the test values so it's not tested yet
    chassis = Chassis()
    
    configMat = []
    for i in range(100):
        curOdom, curArmConfig, curWheelConfig = NextState(curConfig, curSpeeds, dt, chassis.F, wLimit)
        configMat.append(np.concatenate([curOdom, curArmConfig, curWheelConfig, [0]]))
        print(f"Current configuration: {configMat[i]}")
        
        curConfig = np.concatenate([curOdom, curArmConfig, curWheelConfig])
        
    with open('testNextState1.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in configMat:
            writer.writerow(np.array(i))
        file.close()
        
def testNextState2():
    # Starting config
    chassisConfig = np.array([0, 0, 0])
    armConfig = np.array([0, 0, 0, 0, 0])
    wheelConfig = np.array([0, 0, 0, 0])
    curConfig = np.concatenate([chassisConfig, armConfig, wheelConfig])
    
    wheelSpeeds = np.array([-10, 10, -10, 10])
    armSpeeds = np.array([0, 0, 0, 0, 0])
    curSpeeds = np.concatenate([wheelSpeeds, armSpeeds])
    dt = 0.01
    wLimit = 10 # Setting just above the test values so it's not tested yet
    chassis = Chassis()
    
    NextState([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], curSpeeds, dt, chassis.F,  wLimit)
    
    configMat = []
    for i in range(100):
        curOdom, curArmConfig, curWheelConfig = NextState(curConfig, curSpeeds, dt, chassis.F,  wLimit)
        configMat.append(np.concatenate([curOdom, curArmConfig, curWheelConfig, [0]]))
        print(f"Current configuration: {configMat[i]}")
        
        curConfig = np.concatenate([curOdom, curArmConfig, curWheelConfig])
        
    with open('testNextState2.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in configMat:
            writer.writerow(np.array(i))
        file.close()

def testNextState3():
    # Starting config
    chassisConfig = np.array([0, 0, 0])
    armConfig = np.array([0, 0, 0, 0, 0])
    wheelConfig = np.array([0, 0, 0, 0])
    curConfig = np.concatenate([chassisConfig, armConfig, wheelConfig])
    
    wheelSpeeds = np.array([-10, 10, 10, -10])
    armSpeeds = np.array([0, 0, 0, 0, 0])
    curSpeeds = np.concatenate([wheelSpeeds, armSpeeds])
    dt = 0.01
    wLimit = 10
    chassis = Chassis()
    
    NextState([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], curSpeeds, dt, chassis.F,  wLimit)
    
    configMat = []
    for i in range(100):
        curOdom, curArmConfig, curWheelConfig = NextState(curConfig, curSpeeds, 
                                                          dt, chassis.F, wLimit)
        configMat.append(np.concatenate([curOdom, curArmConfig, curWheelConfig, [0]]))
        print(f"Current configuration: {configMat[i]}")
        
        curConfig = np.concatenate([curOdom, curArmConfig, curWheelConfig])
        
    with open('testNextState3.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in configMat:
            writer.writerow(np.array(i))
        file.close()
        
def testTrajectoryGenerator():
    # Defining test conditions
    Tse_init = np.array([[1, 0, 0, 0.2932],
                         [0, 1, 0,  0.094],
                         [0, 0, 1, 0.8475],
                         [0, 0, 0,      1]])
    
    Tsc_init = np.array([[ 1, 0,  0,     1],
                         [ 0, 1,  0,     0],
                         [ 0, 0,  1, 0.025],
                         [ 0, 0,  0,     1]])
    
    Tsc_final = np.array([[ 0, 1,  0,     0],
                          [-1, 0,  0,    -1],
                          [ 0, 0,  1, 0.025],
                          [ 0, 0,  0,     1]])
    
    Tce_grasp = np.array([[-0.707106781,  0,  0.707106781, 0],
                          [            0, 1,            0, 0],
                          [-0.707106781,  0, -0.707106781, 0],
                          [            0, 0,            0, 1]])
    
    Tce_standoff = np.array([[-0.707106781,  0,  0.707106781,    0],
                             [            0, 1,            0,    0],
                             [-0.707106781,  0, -0.707106781, 0.25],
                             [            0, 0,            0,    1]])
    
    k = 1
    
    # Running the function
    trajMat = TrajectoryGenerator(Tse_init, Tsc_init, Tsc_final, Tce_grasp, Tce_standoff, k)
    
    #Outputting the trajectory into a csv
    with open('testTrajectoryGenerator.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in trajMat:
            writer.writerow(np.array(i))
        file.close()
        
def testFeedbackControl():
    # Defining test conditions
    curConfig = np.array([0, 0, 0, 0, 0, 0.2, -1.6, 0, 0, 0, 0, 0])
    
    Xd = np.array([[ 0, 0, 1, 0.5],
                   [ 0, 1, 0,   0],
                   [-1, 0, 0, 0.5],
                   [ 0, 0, 0,   1]])
    
    Xd_n = np.array([[ 0, 0, 1, 0.6],
                     [ 0, 1, 0,   0],
                     [-1, 0, 0, 0.3],
                     [ 0, 0, 0,   1]])
    
    X = np.array([[ 0.170, 0, 0.985, 0.387],
                  [     0, 1,     0,     0],
                  [-0.985, 0, 0.170, 0.570],
                  [     0, 0,     0,     1]])
    
    Kp = np.eye(6) * 0.0
    Ki = np.eye(6) * 0.0
    dt = 0.01
    chassis = Chassis()
    arm = Arm()
    
    intErrout, velCmds, TerrVec = FeedbackControl(curConfig, X, Xd, Xd_n, Kp, Ki, dt, 0,
                                                  arm, chassis)
    print(intErrout)
    print(velCmds)
    print(TerrVec)
        
def createCurNextT(curTrajMat, nextTrajMat):
    Tsed_cur = np.array([[curTrajMat[0], curTrajMat[1], curTrajMat[2],  curTrajMat[9]],
                         [curTrajMat[3], curTrajMat[4], curTrajMat[5], curTrajMat[10]],
                         [curTrajMat[6], curTrajMat[7], curTrajMat[8], curTrajMat[11]],
                         [            0,             0,             0,              1]])
    Tsed_next = np.array([[nextTrajMat[0], nextTrajMat[1], nextTrajMat[2],  nextTrajMat[9]],
                          [nextTrajMat[3], nextTrajMat[4], nextTrajMat[5], nextTrajMat[10]],
                          [nextTrajMat[6], nextTrajMat[7], nextTrajMat[8], nextTrajMat[11]],
                          [             0,              0,              0,               1]])

    return Tsed_cur, Tsed_next
        
def generateBestTraj():
    # # Defining test conditions
    Tse_init = np.array([[1, 0, 0, 0.2932],
                         [0, 1, 0,  0.094],
                         [0, 0, 1, 0.8475],
                         [0, 0, 0,      1]])
    
    Tsc_init = np.array([[ 1, 0,  0,     1],
                         [ 0, 1,  0,     0],
                         [ 0, 0,  1, 0.025],
                         [ 0, 0,  0,     1]])
    
    Tsc_final = np.array([[ 0, 1,  0,     0],
                          [-1, 0,  0,    -1],
                          [ 0, 0,  1, 0.025],
                          [ 0, 0,  0,     1]])
    
    Tce_grasp = np.array([[-0.707106781,  0,  0.707106781, 0],
                          [            0, 1,            0, 0],
                          [-0.707106781,  0, -0.707106781, 0],
                          [            0, 0,            0, 1]])
    
    Tce_standoff = np.array([[-0.707106781,  0,  0.707106781,    0],
                             [            0, 1,            0,    0],
                             [-0.707106781,  0, -0.707106781, 0.25],
                             [            0, 0,            0,    1]])
    
    k = 1
    
    arm = Arm()
    chassis = Chassis()
    
    # Generating the perfect trajectory
    trajMat = TrajectoryGenerator(Tse_init, Tsc_init, Tsc_final, Tce_grasp, Tce_standoff, k)

    # Initializing variables
    Tse_cur = Tse_init
    Kp = np.eye(6) * 1.0
    Ki = np.eye(6) * 0.0
    
    # Initializing vectors
    curConfig = np.array([0, 0.1, 0.1,
                          0.1, 0.2, 0.3, 0.4, 0.5,
                          0.0, 0.0, 0.0, 0.0,
                          0.0])
    configMat = np.array(curConfig)
    errorArray = np.zeros((6, 1))
    curIntErr = errorArray[0]
    
    # Loop that generates trajectories
    for i in range(len(trajMat)-2):                        
        # Calculating Tsbcur
        Tsbcur = np.array([[np.cos(curConfig[0]), -np.sin(curConfig[0]), 0, curConfig[1]],
                           [np.sin(curConfig[0]),  np.cos(curConfig[0]), 0, curConfig[2]],
                           [                   0,                     0, 1,       0.0963],
                           [                   0,                     0, 0,            1]])
        
        # Create a transformation matrix based on the current trajMat
        Tsed_cur, Tsed_next = createCurNextT(trajMat[i], trajMat[i+1])
        
        # Calculate the control law
        curIntErr, curSpeeds, curErrVec = FeedbackControl(curConfig, Tse_cur, Tsed_cur,
                                                          Tsed_next, Kp, Ki, 0.01, curIntErr,
                                                          arm, chassis)
        
        # Sending data to NextState
        newOdom, newArmConfig, newWheelConfig = NextState(curConfig, curSpeeds, 0.01,
                                                          chassis.F)
        
        # Making the current variables = the new variables
        curConfig = np.concatenate([newOdom, newArmConfig, newWheelConfig, [trajMat[i][12]]])
        Tse_cur = Tsbcur @ chassis.Tb0 @ mr.FKinBody(arm.Mhome, arm.Blist, newArmConfig)
        
        # Storing every kth state
        if i % k == 0:
            # Store this state
            configMat = np.block([[configMat], [curConfig]])
            # Adding the current error vector to the array of error vectors
            errorArray = np.block([errorArray, np.array([curErrVec]).T])
        
    #Outputting the trajectory into a csv
    print("Writing trajectory to csv file")
    with open('BestTrajectory.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in configMat:
            writer.writerow(np.array(i))
        file.close()
        
    #Outputting the error into a csv
    print("Writing error to csv file")
    with open('BestError.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in errorArray.T:
            writer.writerow(np.array(i))
        file.close()
        
    # Create error trajectory
    print("Creating error plot")
    dt = 0.01
    tvec = np.arange(0, len(errorArray[0])*dt, dt)
    line1, = plt.plot(tvec, errorArray[0], label='roll')
    line2, = plt.plot(tvec, errorArray[1], label='pitch')
    line3, = plt.plot(tvec, errorArray[2], label='yaw')
    line4, = plt.plot(tvec, errorArray[3], label='x')
    line5, = plt.plot(tvec, errorArray[4], label='y')
    line6, = plt.plot(tvec, errorArray[5], label='z')
    plt.title(f"Error Vector vs. Time\n[Best]")
    plt.xlabel('Time (s)')
    plt.ylabel('Error (rad and m)')
    plt.legend(loc=4)
    plt.show()
    
    # DONE!!
    print("Finshed")
    
def generateOvershootTraj():
    # Defining initial conditions
    Tse_init = np.array([[1, 0, 0, 0.2932],
                         [0, 1, 0,  0.094],
                         [0, 0, 1, 0.8475],
                         [0, 0, 0,      1]])
    
    Tsc_init = np.array([[ 1, 0,  0,     1],
                         [ 0, 1,  0,     0],
                         [ 0, 0,  1, 0.025],
                         [ 0, 0,  0,     1]])
    
    Tsc_final = np.array([[ 0, 1,  0,     0],
                          [-1, 0,  0,    -1],
                          [ 0, 0,  1, 0.025],
                          [ 0, 0,  0,     1]])
    
    Tce_grasp = np.array([[-0.707106781,  0,  0.707106781, 0],
                          [            0, 1,            0, 0],
                          [-0.707106781,  0, -0.707106781, 0],
                          [            0, 0,            0, 1]])
    
    Tce_standoff = np.array([[-0.707106781,  0,  0.707106781,    0],
                             [            0, 1,            0,    0],
                             [-0.707106781,  0, -0.707106781, 0.25],
                             [            0, 0,            0,    1]])
    
    k = 1
    
    arm = Arm()
    chassis = Chassis()
    
    # Generating the perfect trajectory
    trajMat = TrajectoryGenerator(Tse_init, Tsc_init, Tsc_final, Tce_grasp, Tce_standoff, k)
    
    # Initializing variables
    Tse_cur = Tse_init
    Kp = np.eye(6) * 1.0
    Ki = np.eye(6) * 0.05
    
    # Initializing vectors
    curConfig = np.array([0, 0.0, 0.2,
                          0.0, 0.6, 0.0, 0.0, 0.0,
                          0.0, 0.0, 0.0, 0.0,
                          0.0])
    configMat = np.array(curConfig)
    errorArray = np.zeros((6, 1))
    curIntErr = errorArray[0]
    
    # Loop that generates trajectories
    for i in range(len(trajMat)-2):
        # Calculating Tsbcur
        Tsbcur = np.array([[np.cos(curConfig[0]), -np.sin(curConfig[0]), 0, curConfig[1]],
                           [np.sin(curConfig[0]),  np.cos(curConfig[0]), 0, curConfig[2]],
                           [                   0,                     0, 1,       0.0963],
                           [                   0,                     0, 0,            1]])
        
        # Create a transformation matrix based on the current trajMat
        Tsed_cur, Tsed_next = createCurNextT(trajMat[i], trajMat[i+1])
        
        # Calculate the control law
        curIntErr, curSpeeds, curErrVec = FeedbackControl(curConfig, Tse_cur, Tsed_cur,
                                                          Tsed_next, Kp, Ki, 0.01, curIntErr,
                                                          arm, chassis)
        
        # Sending data to NextState
        newOdom, newArmConfig, newWheelConfig = NextState(curConfig, curSpeeds, 0.01,
                                                          chassis.F)
        
        # Making the current variables = the new variables
        curConfig = np.concatenate([newOdom, newArmConfig, newWheelConfig, [trajMat[i][12]]])
        Tse_cur = Tsbcur @ chassis.Tb0 @ mr.FKinBody(arm.Mhome, arm.Blist, newArmConfig)
        
        # Storing every kth state
        if i % k == 0:
            # Store this state
            configMat = np.block([[configMat], [curConfig]])
            # Adding the current error vector to the array of error vectors
            errorArray = np.block([errorArray, np.array([curErrVec]).T])
        
    #Outputting the trajectory into a csv
    print("Writing to csv file")
    with open('OvershootTrajectory.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in configMat:
            writer.writerow(np.array(i))
        file.close()
        
    #Outputting the error into a csv
    print("Writing error to csv file")
    with open('OvershootError.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in errorArray.T:
            writer.writerow(np.array(i))
        file.close()
        
    # Create error trajectory
    print("Creating error plot")
    dt = 0.01
    tvec = np.arange(0, len(errorArray[0])*dt, dt)
    line1, = plt.plot(tvec, errorArray[0], label='roll')
    line2, = plt.plot(tvec, errorArray[1], label='pitch')
    line3, = plt.plot(tvec, errorArray[2], label='yaw')
    line4, = plt.plot(tvec, errorArray[3], label='x')
    line5, = plt.plot(tvec, errorArray[4], label='y')
    line6, = plt.plot(tvec, errorArray[5], label='z')
    plt.title(f"Error Vector vs. Time\n[Overshoot]")
    plt.xlabel('Time (s)')
    plt.ylabel('Error (rad and m)')
    plt.legend(loc=4)
    plt.show()
    
    # DONE!!
    print("Finshed")
       
def generateNewTaskTraj():
    # # Defining test conditions
    Tse_init = np.array([[1, 0, 0, 0.2932],
                         [0, 1, 0,  0.094],
                         [0, 0, 1, 0.8475],
                         [0, 0, 0,      1]])
    
    Tsc_init = np.array([[ 1, 0,  0,   0.5],
                         [ 0, 1,  0,   0.5],
                         [ 0, 0,  1, 0.025],
                         [ 0, 0,  0,     1]])
    
    Tsc_final = np.array([[ 0, 1,  0,  -0.5],
                          [-1, 0,  0,    -1],
                          [ 0, 0,  1, 0.025],
                          [ 0, 0,  0,     1]])
    
    Tce_grasp = np.array([[-0.707106781,  0,  0.707106781, 0],
                          [            0, 1,            0, 0],
                          [-0.707106781,  0, -0.707106781, 0],
                          [            0, 0,            0, 1]])
    
    Tce_standoff = np.array([[-0.707106781,  0,  0.707106781,    0],
                             [            0, 1,            0,    0],
                             [-0.707106781,  0, -0.707106781, 0.25],
                             [            0, 0,            0,    1]])
    
    k = 1
    
    arm = Arm()
    chassis = Chassis()
    
    # Generating the perfect trajectory
    trajMat = TrajectoryGenerator(Tse_init, Tsc_init, Tsc_final, Tce_grasp, Tce_standoff, k)

    # Initializing variables
    Tse_cur = Tse_init
    Kp = np.eye(6) * 1.0
    Ki = np.eye(6) * 0.0
    
    # Initializing vectors
    curConfig = np.array([0, 0.1, 0.1,
                          0.1, 0.2, 0.3, 0.4, 0.5,
                          0.0, 0.0, 0.0, 0.0,
                          0.0])
    configMat = np.array(curConfig)
    errorArray = np.zeros((6, 1))
    curIntErr = errorArray[0]
    
    # Loop that generates trajectories
    for i in range(len(trajMat)-2):                        
        # Calculating Tsbcur
        Tsbcur = np.array([[np.cos(curConfig[0]), -np.sin(curConfig[0]), 0, curConfig[1]],
                           [np.sin(curConfig[0]),  np.cos(curConfig[0]), 0, curConfig[2]],
                           [                   0,                     0, 1,       0.0963],
                           [                   0,                     0, 0,            1]])
        
        # Create a transformation matrix based on the current trajMat
        Tsed_cur, Tsed_next = createCurNextT(trajMat[i], trajMat[i+1])
        
        # Calculate the control law
        curIntErr, curSpeeds, curErrVec = FeedbackControl(curConfig, Tse_cur, Tsed_cur,
                                                          Tsed_next, Kp, Ki, 0.01, curIntErr,
                                                          arm, chassis)
        
        # Sending data to NextState
        newOdom, newArmConfig, newWheelConfig = NextState(curConfig, curSpeeds, 0.01,
                                                          chassis.F)
        
        # Making the current variables = the new variables
        curConfig = np.concatenate([newOdom, newArmConfig, newWheelConfig, [trajMat[i][12]]])
        Tse_cur = Tsbcur @ chassis.Tb0 @ mr.FKinBody(arm.Mhome, arm.Blist, newArmConfig)
        
        # Storing every kth state
        if i % k == 0:
            # Store this state
            configMat = np.block([[configMat], [curConfig]])
            # Adding the current error vector to the array of error vectors
            errorArray = np.block([errorArray, np.array([curErrVec]).T])
        
    #Outputting the trajectory into a csv
    print("Writing trajectory to csv file")
    with open('NewTaskTrajectory.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in configMat:
            writer.writerow(np.array(i))
        file.close()
        
    #Outputting the error into a csv
    print("Writing error to csv file")
    with open('NewTaskError.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in errorArray.T:
            writer.writerow(np.array(i))
        file.close()
        
    # Create error trajectory
    print("Creating error plot")
    dt = 0.01
    tvec = np.arange(0, len(errorArray[0])*dt, dt)
    line1, = plt.plot(tvec, errorArray[0], label='roll')
    line2, = plt.plot(tvec, errorArray[1], label='pitch')
    line3, = plt.plot(tvec, errorArray[2], label='yaw')
    line4, = plt.plot(tvec, errorArray[3], label='x')
    line5, = plt.plot(tvec, errorArray[4], label='y')
    line6, = plt.plot(tvec, errorArray[5], label='z')
    plt.title(f"Error Vector vs. Time\n[New Task]")
    plt.xlabel('Time (s)')
    plt.ylabel('Error (rad and m)')
    plt.legend(loc=4)
    plt.show()
    
    # DONE!!
    print("Finshed")
          
def main(args=None):
    generateBestTraj()
    generateOvershootTraj()
    generateNewTaskTraj()
    
main()