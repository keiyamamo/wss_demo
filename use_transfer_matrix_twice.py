from dolfin import *
import numpy as np

# ------ epsilon(u) and class STRESS are direct copies from VaMPy ---------
def epsilon(u):
    """
    Computes the strain-rate tensor
    Args:
        u (Function): Velocity field

    Returns:
        epsilon (Function): Strain rate tensor of u
    """

    return 0.5 * (grad(u) + grad(u).T)


class STRESS:
    def __init__(self, u, p, nu, mesh):
        boundary_ds = Measure("ds", domain=mesh)
        boundary_mesh = BoundaryMesh(mesh, 'exterior')
        self.bmV = VectorFunctionSpace(boundary_mesh, 'CG', 1)

        # Compute stress tensor
        sigma = (2 * nu * epsilon(u)) - (p * Identity(len(u)))
        # Compute stress on surface
        n = FacetNormal(mesh)
        F = -(sigma * n)

        # Compute normal and tangential components
        Fn = inner(F, n)  # scalar-valued
        Ft = F - (Fn * n)  # vector-valued

        # Integrate against piecewise constants on the boundary
        scalar = FunctionSpace(mesh, 'DG', 0)
        vector = VectorFunctionSpace(mesh, 'CG', 1)
        scaling = FacetArea(mesh)  # Normalise the computed stress relative to the size of the element

        v = TestFunction(scalar)
        w = TestFunction(vector)

        # Create functions
        self.Fn = Function(scalar)
        self.Ftv = Function(vector)
        self.Ft = Function(scalar)

        self.Ln = 1 / scaling * v * Fn * boundary_ds
        self.Ltv = 1 / (2 * scaling) * inner(w, Ft) * boundary_ds
        self.Lt = 1 / scaling * inner(v, self.norm_l2(self.Ftv)) * boundary_ds

    def __call__(self):
        """
        Compute stress for given velocity field u and pressure field p

        Returns:
            Ftv_mb (Function): Shear stress
        """

        # Assemble vectors
        assemble(self.Ltv, tensor=self.Ftv.vector())
        self.Ftv_bm = interpolate(self.Ftv, self.bmV)

        return self.Ftv_bm

    def norm_l2(self, u):
        """
        Compute norm of vector u in expression form
        Args:
            u (Function): Function to compute norm of

        Returns:
            norm (Power): Norm as expression
        """
        return pow(inner(u, u), 0.5)

# ------ epsilon(u) and class STRESS are direct copies from VaMPy ---------

def main():
    # create a mesh and refined mesh
    mesh = UnitCubeMesh(10, 10, 10)
    refined_mesh = refine(mesh)

    # create function spaces for the two meshes
    V2 = VectorFunctionSpace(mesh, "CG", 2)
    V1 = VectorFunctionSpace(refined_mesh, "CG", 1)

    # genereate some function as an example
    f = Expression(("sin(x[0]*pi)", "cos(x[1]*pi)", "x[2]"), degree=2)

    # First, interpolate the function onto the higher order function space
    u_2 = interpolate(f, V2)
    # transfer matrix here is P2 --> P1
    transfer_matrix = PETScDMCollection.create_transfer_matrix(V2, V1)

    # create a function on the lower order function space
    u_1 = Function(V1)
    # Interpolate the function onto the lower order function space
    u_1.vector()[:] = transfer_matrix * u_2.vector()

    # compute pseudo-stress on the lower order function space
    stress = STRESS(u=u_1, p=0.0, nu=1.0, mesh=refined_mesh)
    tau = stress()

    # Now, we will re-interpolate the function onto the higher order function space
    back_transfer_matrix = PETScDMCollection.create_transfer_matrix(V1, V2)

    # create a function on the higher order function space
    u_2_back = Function(V2)

    # Interpolate the function onto the higher order function space
    u_2_back.vector()[:] = back_transfer_matrix * u_1.vector()

    # Again, compute the pseudo-stress on the higher order function space
    stress_2 = STRESS(u=u_2_back, p=0.0, nu=1.0, mesh=mesh)
    tau_2 = stress_2()

    File("tau.pvd") << tau
    File("tau_2.pvd") << tau_2

if __name__ == '__main__':
    main()