import numpy as np
import g2o

class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

        # cam = g2o.CameraParameters(1000, (320, 240), 0)
        # cam.set_id(0)
        # super().add_parameter(cam)
        # initial_position=np.zeros(3), 
        # initial_orientation=g2o.Quaternion()

    def optimize(self, max_iterations=20):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_vertex(self, id, pose, fixed=False):
        v_se3 = g2o.VertexSE3()
        # v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_id(id)
        v_se3.set_estimate(pose)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)


    # def add_edge(self, vertices, measurement, 
    #         information=np.identity(2),
    #         robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991))): #95% CI
        
    #     edge = g2o.EdgeSE3()
    #     #edge = g2o.EdgeProjectXYZ2UV
    #     for i, v in enumerate(vertices):
    #         if isinstance(v, int):
    #             v = self.vertex(v)
    #         edge.set_vertex(i, v)

    #     edge.set_measurement(measurement)  # relative pose
    #     edge.set_information(information)

    #     # robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))
    #     if robust_kernel is not None:
    #         edge.set_robust_kernel(robust_kernel)
    #     super().add_edge(edge)

    def get_pose(self, id):
        return self.vertex(id).estimate()

    def get_vertices(self):
        return super().vertices()
        # print(vertices)
        # print(type(vertices))
        # print(vertices[2])
        
    def add_edge(self, v0,v1, measurement=None, 
            information=np.identity(6),
            robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991))): #95% CI
        
        edge = g2o.EdgeSE3()
        # edge = g2o.EdgeProjectXYZ2UV()

        edge.set_vertex(0, v0)
        edge.set_vertex(1, v1)
        if measurement is None:
            measurement = g2o.Isometry3d(g2o.Quaternion(0,0,0,1),np.array([0,1,0]))
        edge.set_measurement(measurement)  # relative pose
        edge.set_information(information)
        # edge.set_parameter_id(0, 0)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)


    def test_add_edge(self,edge):
        super().add_edge(edge)

if __name__ == "__main__":
    pose0 = np.array([0,0,0]) #+ np.random.normal(0,0.1,3)
    pose1 = np.array([0,1,0]) + np.random.normal(0,0.1,3)
    pose2 = np.array([0,2,0]) + np.random.normal(0,0.1,3)
    print("poses:",'\n',pose0,'\n',pose1,'\n',pose2,'\n')
    opt = PoseGraphOptimization()
    # pose0 = g2o.SE3Quat(np.identity(6), pose0)
    # pose1 = g2o.SE3Quat(np.identity(6), pose1)
    # pose2 = g2o.SE3Quat(np.identity(6), pose2)
    pose0 = g2o.Isometry3d(g2o.Quaternion(0,0,0,1),pose0)
    pose1 = g2o.Isometry3d(g2o.Quaternion(0,0,0,1),pose1)
    pose2 = g2o.Isometry3d(g2o.Quaternion(0,0,0,1),pose2)

    opt.add_vertex(0,pose0,True)
    opt.add_vertex(1,pose1)
    opt.add_vertex(2,pose2)
    
    vertices = opt.get_vertices()

    opt.add_edge(vertices[0],vertices[1],measurement=g2o.Isometry3d(g2o.Quaternion(0,0,0,1),np.array([0,-1,0])))
    opt.add_edge(vertices[1],vertices[2],measurement=g2o.Isometry3d(g2o.Quaternion(0,0,0,1),np.array([0,-1,0])))
    opt.add_edge(vertices[0],vertices[2],measurement=g2o.Isometry3d(g2o.Quaternion(0,0,0,1),np.array([0,-2,0])))

    opt.optimize(2)

    if opt.vertex(0).fixed():
        print('First fixed')
    
    print(opt.get_pose(0).translation())
    print(opt.get_pose(1).translation())
    print(opt.get_pose(2).translation())

    opt.optimize(5)

    print(opt.get_pose(0).translation())
    print(opt.get_pose(1).translation())
    print(opt.get_pose(2).translation())
    # print(estimated.rotation())

    
    # print(estimated(0,0),estimated(0,1),estimated(0,2),estimated(0,3),estimated(0,4),estimated(0,5),estimated(0,6),estimated(0,0))

    # print(opt.)
    # edge = g2o.EdgeSE3()
    # edge = g2o.EdgeProjectXYZ2UV()

    # edge.set_vertex(0, vertices[0])
    # edge.set_vertex(1, vertices[1])
    # measurement = np.array([0,1])
    # information = np.identity(2)
    # edge.set_measurement(measurement)  # relative pose
    # edge.set_information(information)
    # robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991))
    # edge.set_robust_kernel(robust_kernel)
    # opt.test_add_edge(edge)
