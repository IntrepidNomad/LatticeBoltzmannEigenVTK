#ifndef INITIALIZATION_H
#define INITIALIZATION_H

#endif // INITIALIZATION_H

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Core>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <algorithm>
#include <random>

struct params {
    const static int Nx = 400; //resolution x-dir
    const static int Ny = 100; //resolution y-dir
    const double rho0 = 100; // average density
    double tau = 0.8; // collision timescale
    const int Nt = 6000; // number of timesteps

};

struct lattice {
    const static int NL = 9; // number of nodes

    // idxs: Index array representing the 9 possible directions in the D2Q9 model
    const Eigen::Vector<int,NL> idxs = Eigen::Vector<int,9>::LinSpaced(9,0,8);

    // cxs: Vector representing the x-components of the 9 discrete velocities in the D2Q9 model
    // (0,0) represents zero velocity,
    // (1,0), (0,1), (-1,0), (0,-1) are the cardinal directions (East, North, West, South),
    // (1,1), (-1,1), (-1,-1), (1,-1) are the diagonal directions.
    const Eigen::Vector<int,NL> cxs = {0, 0, 1, 1, 1, 0,-1,-1,-1};

    // cys: Vector representing the y-components of the 9 discrete velocities in the D2Q9 model
    // The arrangement corresponds to the velocities defined in cxs.
    const Eigen::Vector<int,NL> cys = {0, 1, 1, 0,-1,-1,-1, 0, 1};

    // weights: Weights associated with each of the 9 directions in the D2Q9 model
    // These weights are used in the computation of equilibrium distribution functions
    // and reflect the isotropy of the model.
    // The central weight (for zero velocity) is 4/9, cardinal directions are 1/9,
    // and diagonals are 1/36.
    const Eigen::Vector<double,NL> weights = {4.0/9,1.0/9,1.0/36,1.0/9,1.0/36,1.0/9,1.0/36,1.0/9,1.0/36};
};

Eigen::Tensor<double, 3> initializeF(const params& simulationParams) {
    Eigen::Tensor<double, 3> F(simulationParams.Nx, simulationParams.Ny, 9);

    // Initialize F with uniform distribution (e.g., rho0 / NL)
    F.setConstant(1.8);//simulationParams.rho0/9);

    // Add small random perturbations to F
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 0.02);
    for (int x = 0; x < simulationParams.Nx; ++x) {
        for (int y = 0; y < simulationParams.Ny; ++y) {
            for (int i = 0; i < 9; ++i) {
                F(x, y, i) += distribution(generator);
            }
        }
    }

    return F;
}


template <typename T>
Eigen::Tensor<T, 2> roll2D(const Eigen::Tensor<T, 2>& tensor, int shift, int axis) {
    int Nx = tensor.dimension(0);
    int Ny = tensor.dimension(1);

    Eigen::Tensor<T, 2> result(tensor.dimensions());

    if (axis == 0) {  // Roll along x-axis
        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                int rolled_x = (x + shift) % Nx;
                if (rolled_x < 0) rolled_x += Nx;
                result(rolled_x, y) = tensor(x, y);
            }
        }
    } else if (axis == 1) {  // Roll along y-axis
        for (int y = 0; y < Ny; ++y) {
            for (int x = 0; x < Nx; ++x) {
                int rolled_y = (y + shift) % Ny;
                if (rolled_y < 0) rolled_y += Ny;
                result(x, rolled_y) = tensor(x, y);
            }
        }
    }

    return result;
}
