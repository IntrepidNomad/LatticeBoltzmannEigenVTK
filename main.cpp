#include <initialization.h>

//VTK Includes
#include <vtkImageActor.h>
#include <vtkImageData.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkCellArray.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkCubeSource.h>
#include <vtkAutoInit.h>
#include <vtkColorTransferFunction.h>
#include <vtkImageMapToColors.h>
#include <vtkSmartPointer.h>
#include <vtkCamera.h>


VTK_MODULE_INIT(vtkRenderingOpenGL2);
VTK_MODULE_INIT(vtkInteractionStyle);

vtkSmartPointer<vtkImageData> imageData = vtkSmartPointer<vtkImageData>::New();
// Configure imageData with dimensions and fill it with vorticity data

vtkSmartPointer<vtkImageActor> actor = vtkSmartPointer<vtkImageActor>::New();

vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();

vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();

vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor = vtkSmartPointer<vtkRenderWindowInteractor>::New();

vtkSmartPointer<vtkColorTransferFunction> colorFunc = vtkSmartPointer<vtkColorTransferFunction>::New();

vtkSmartPointer<vtkImageMapToColors> colorMap = vtkSmartPointer<vtkImageMapToColors>::New();
// Define RGB values for darker teal and subdued yellow
double darkerTeal[3] = {0.0, 0.5, 0.5};  // Example RGB values for darker teal
double subduedYellow[3] = {0.8, 0.8, 0.4};  // Example RGB values for subdued yellow

vtkSmartPointer<vtkCamera> camera = vtkSmartPointer<vtkCamera>::New();


using namespace std;

template <typename T>
Eigen::Tensor<T, 3> roll(const Eigen::Tensor<T, 3>& tensor, int shift, int axis) {
    int Nx = tensor.dimension(0);
    int Ny = tensor.dimension(1);
    int NL = tensor.dimension(2);

    Eigen::Tensor<T, 3> result(tensor.dimensions());

    if (axis == 0) {  // Roll along Nx (x-axis)
        for (int l = 0; l < NL; ++l) {
            for (int y = 0; y < Ny; ++y) {
                for (int x = 0; x < Nx; ++x) {
                    int rolled_x = (x + shift) % Nx;
                    if (rolled_x < 0) rolled_x += Nx;
                    result(rolled_x, y, l) = tensor(x, y, l);
                }
            }
        }
    } else if (axis == 1) {  // Roll along Ny (y-axis)
        for (int l = 0; l < NL; ++l) {
            for (int y = 0; y < Ny; ++y) {
                for (int x = 0; x < Nx; ++x) {
                    int rolled_y = (y + shift) % Ny;
                    if (rolled_y < 0) rolled_y += Ny;
                    result(x, rolled_y, l) = tensor(x, y, l);
                }
            }
        }
    }

    return result;
}

double cubicRootScaling(double value) {
    if (value < 0) {
        return -std::pow(-value, 1.0 / 3.0);
    } else {
        return std::pow(value, 1.0 / 3.0);
    }
}


int main()
{

    // Map vorticity values to colors
    colorFunc->AddRGBPoint(-1.0, darkerTeal[0], darkerTeal[1], darkerTeal[2]); // Darker teal for low vorticity
    colorFunc->AddRGBPoint(0.0, 0.4, 0.2, 0.4);  // Mid-value (e.g., gray)
    colorFunc->AddRGBPoint(1.0, subduedYellow[0], subduedYellow[1], subduedYellow[2]); // Subdued yellow for high vorticity

    // Create the vtkImageMapToColors filter
    // Create an instance of your parameter struct
    params simulationParams;
    lattice simulationLattice;

    colorMap->SetInputData(imageData); // imageData contains the vorticity values
    colorMap->SetLookupTable(colorFunc); // colorFunc is your vtkColorTransferFunction

    imageData->SetDimensions(simulationParams.Nx, simulationParams.Ny, 1);
    imageData->AllocateScalars(VTK_DOUBLE, 1);
    // Use the output of colorMap in your visualization pipeline

    actor->SetInputData(colorMap->GetOutput()); // Use the colored output
    renderer->AddActor(actor);
    renderWindow->AddRenderer(renderer);
    //renderWindow->FullScreenOn();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    renderWindowInteractor->Initialize();






    Eigen::Tensor<double,3> F = initializeF(simulationParams);


    // Create meshgrid matrices X and Y
    Eigen::Tensor<double,2> X(simulationParams.Nx,simulationParams.Ny);
    Eigen::Tensor<double,2> Y(simulationParams.Nx,simulationParams.Ny);
    for (int i = 0; i < simulationParams.Nx; ++i) {
        for (int j = 0; j < simulationParams.Ny; ++j) {
            X(i, j) = i;
            Y(i, j) = j;
        }
    }

    // Convert X, Y to 3D tensors for broadcasting, reshaped to match F's dimensions
    //Eigen::TensorMap<Eigen::Tensor<const double, 3>> X_tensor(X.data(), simulationParams.Nx, simulationParams.Ny, 1);
    //Eigen::TensorMap<Eigen::Tensor<const double, 3>> Y_tensor(Y.data(), simulationParams.Nx, simulationParams.Ny, 1);

    //Eigen::array<int, 3> bcast_XY = {1, 1, simulationLattice.NL};
    //Eigen::array<int, 3> bcast_rho = {simulationParams.Ny, simulationParams.Nx, 1}; No longer needed.

    // Compute the cosine term using unaryExpr with a lambda function
    //Eigen::Tensor<double, 3> cosineArg = (2 * M_PI * X_tensor.broadcast(bcast_XY) / static_cast<double>(simulationParams.Nx * 4));
    //Eigen::Tensor<double, 3> cosineTerm = cosineArg.unaryExpr([](double x) { return std::cos(x); });
    Eigen::Tensor<double,2> cosineArg = (2 * M_PI * X / (static_cast<double>(simulationParams.Nx * 4)));
    Eigen::Tensor<double,2> cosineTerm = cosineArg.unaryExpr([](double x) {return std::cos(x);});

    // Compute the adjustment and update F
    //Eigen::Tensor<double, 3> ones = F.constant(1.0);
    Eigen::Tensor<double,2> ones(simulationParams.Nx,simulationParams.Ny);
    ones.setConstant(1.0);
    //Eigen::Tensor<double, 3> adjustment = (2 * (ones + 0.2 * cosineTerm));
    Eigen::Tensor<double,2> adjustment = (4 * (ones + 0.6 * cosineTerm));
    F.chip(3,2) += adjustment;

    // Sum along the third dimension to get rho
    Eigen::Tensor<double, 2> rho = F.sum(Eigen::array<int, 1>{2});

    // Reshape rho for broadcasting to match the dimensions of F
    Eigen::Tensor<double, 3> rho_reshaped = rho.reshape(Eigen::array<int, 3>{simulationParams.Nx, simulationParams.Ny, 1})
                                                .broadcast(Eigen::array<int, 3>{1, 1, simulationLattice.NL});

    // Normalize F by multiplying with rho0 and dividing by rho
    F = F * (simulationParams.rho0 / rho_reshaped);

    Eigen::Tensor<int,2> cylinder(simulationParams.Nx, simulationParams.Ny);

    for (int x = 0; x < simulationParams.Nx; ++x) {
        for (int y = 0; y < simulationParams.Ny; ++y) {
            // Compute the squared distance from the center of the cylinder
            double dist_squared = std::pow(x - simulationParams.Nx / 4.0, 2) + std::pow(y - simulationParams.Ny / 2.0, 2);

            // Check if the point is inside the cylinder
            if (dist_squared < std::pow(simulationParams.Ny / 4.0, 2)){
                cylinder(x,y) = 1;
            } else {
                cylinder(x,y) = 0;
            }
        }
    }

    for (int it = 0; it < simulationParams.Nt; ++it) {
            // Drift (Streaming) Step
            // Custom implementation of data shifting
        for (int i = 0; i < simulationLattice.idxs.size(); ++i) {
            int cx = simulationLattice.cxs[i];
            int cy = simulationLattice.cys[i];

            // Roll along x-axis
            if (cx != 0) {
                Eigen::Tensor<double,2> sliceX = F.chip<2>(simulationLattice.idxs[i]).eval();  // Materialize to a 2D Tensor
                F.chip<2>(simulationLattice.idxs[i]) = roll2D(sliceX, cx, 0);
            }

            // Roll along y-axis
            if (cy != 0) {
                Eigen::Tensor<double,2> sliceY = F.chip<2>(simulationLattice.idxs[i]).eval();  // Materialize to a 2D Tensor
                F.chip<2>(simulationLattice.idxs[i]) = roll2D(sliceY, cy, 1);
            }
        }


        // Create a tensor for boundary values, same size as F
        Eigen::Tensor<double, 3> bndryF = F;  // Initially, copy F

        // The new order of the layers
        std::vector<int> newOrder = {0, 5, 6, 7, 8, 1, 2, 3, 4};

        // Extract and store boundary values
        for (int x = 0; x < F.dimension(0); ++x) {
            for (int y = 0; y < F.dimension(1); ++y) {
                if (cylinder(x, y) == 1) {  // Check if the point is part of the boundary
                    for (int z = 0; z < F.dimension(2); ++z) {
                        bndryF(x, y, z) = F(x, y, newOrder[z]);  // Apply reordering
                    }
                }
            }
        }


            // Calculate rho, ux, uy
            // Use Eigen operations for reductions and calculations

        Eigen::Tensor<double, 2> rho = F.sum(Eigen::array<int, 1>({2}));

        Eigen::Tensor<double, 2> ux(F.dimension(0), F.dimension(1));
        Eigen::Tensor<double, 2> uy(F.dimension(0), F.dimension(1));
        ux.setZero(); // Initialize to zero
        uy.setZero(); // Initialize to zero

        for (int z = 0; z < F.dimension(2); ++z) {
            double cx = static_cast<double>(simulationLattice.cxs[z]);
            double cy = static_cast<double>(simulationLattice.cys[z]);

            for (int y = 0; y < F.dimension(1); ++y) {
                for (int x = 0; x < F.dimension(0); ++x) {
                    ux(x, y) += F(x, y, z) * cx;
                    uy(x, y) += F(x, y, z) * cy;
                }
            }
        }



        // Define a small epsilon value
        const double epsilon = std::numeric_limits<double>::epsilon();

        // Create a tensor that flags where rho is zero
        Eigen::Tensor<double, 2> rho_is_zero = rho.unaryExpr([](double r) { return r == 0 ? 1.0 : 0.0; });

        // Replace zeros in rho with epsilon
        Eigen::Tensor<double, 2> safe_rho = rho + (rho_is_zero * epsilon);

        // Calculate ux and uy
        ux = ux / safe_rho;
        uy = uy / safe_rho;

            // Collision Step
            // Compute Feq and update F
        Eigen::Tensor<double, 3> Feq(F.dimensions());
        double tau = simulationParams.tau;

        for (int z = 0; z < F.dimension(2); ++z) {
            double w = simulationLattice.weights[z];
            double cx = static_cast<double>(simulationLattice.cxs[z]);
            double cy = static_cast<double>(simulationLattice.cys[z]);

            for (int y = 0; y < F.dimension(1); ++y) {
                for (int x = 0; x < F.dimension(0); ++x) {
                    double uxcx = ux(x, y) * cx;
                    double uycy = uy(x, y) * cy;
                    Feq(x, y, z) = rho(x, y) * w * (1 + 3 * (uxcx + uycy) + 4.5 * std::pow(uxcx + uycy, 2) - 1.5 * (std::pow(ux(x, y), 2) + std::pow(uy(x, y), 2)));
                }
            }
        }

        // Collision step
        F +=  -(1.0 / tau) * (F - Feq);

            // Apply Boundary Conditions again
            // Reapply reflective boundaries
        for (int x = 0; x < cylinder.dimension(0); ++x) {
            for (int y = 0; y < cylinder.dimension(1); ++y) {
                if (cylinder(x, y) == 1) {  // Assuming cylinder matrix is filled with 0s and 1s
                    for (int z = 0; z < F.dimension(2); ++z) {
                        F(x, y, z) = bndryF(x, y, z);
                    }
                }
            }
        }

        // Set velocity to zero at the boundaries
        for (int x = 0; x < cylinder.dimension(0); ++x) {
            for (int y = 0; y < cylinder.dimension(1); ++y) {
                if (cylinder(x, y)==1) {
                    ux(x, y) = 0;
                    uy(x, y) = 0;
                }
            }
        }

        // Calculate vorticity: vorticity = d(uy)/dx - d(ux)/dy
        Eigen::Tensor<double, 2> vorticity(ux.dimension(0), ux.dimension(1));
        vorticity.setZero();

        for (int y = 0; y < ux.dimension(1); ++y) {
            for (int x = 0; x < ux.dimension(0); ++x) {
                // Apply periodic boundary conditions for the roll operation
                int x_next = (x + 1) % ux.dimension(0);
                int y_next = (y + 1) % ux.dimension(1);

                double dux_dy = ux(x, y_next) - ux(x, y);
                double duy_dx = uy(x_next, y) - uy(x, y);

                vorticity(x, y) = duy_dx - dux_dy;
            }
        }

        // Apply NAN to vorticity at the cylinder locations
        for (int x = 0; x < cylinder.dimension(0); ++x) {
            for (int y = 0; y < cylinder.dimension(1); ++y) {
                if (cylinder(x, y) > 0.5) {
                    vorticity(x, y) = std::numeric_limits<double>::quiet_NaN();
                }
            }
        }
        // Initialize min and max vorticity with extreme values
        double minVorticity = std::numeric_limits<double>::max();
        double maxVorticity = std::numeric_limits<double>::lowest();

        // Manually find min and max values
        for (int y = 0; y < vorticity.dimension(1); ++y) {
            for (int x = 0; x < vorticity.dimension(0); ++x) {
                double value = vorticity(x, y);
                if (value < minVorticity) {
                    minVorticity = value;
                }
                if (value > maxVorticity) {
                    maxVorticity = value;
                }
            }
        }

        // Check the calculated min and max values
        //std::cout << "Min Vorticity: " << minVorticity << ", Max Vorticity: " << maxVorticity << std::endl;

        // Normalize vorticity values to the range [-1, 1] (or any other range you defined in your color transfer function)
        Eigen::Tensor<double, 2> normalizedVorticity = (vorticity - minVorticity) / (maxVorticity - minVorticity) * 2.0 -1.0;
        // Update vtkImageData with normalized vorticity values
        for (int x = 0; x < normalizedVorticity.dimension(0); ++x) {
            for (int y = 0; y < normalizedVorticity.dimension(1); ++y) {
                double value = vorticity(x, y)*30;
                if (std::isnan(value)) {
                    value = 0; // or some other sentinel value
                }

                // Assuming imageData is vtkImageData
                unsigned char color[3];
                // Now you don't need to manually map colors, as the value is in the expected range
                // Remove the manual color mapping here

                imageData->SetScalarComponentFromDouble(x, y, 0, 0, value);
            }
        }
        imageData->Modified();
        colorMap->Update(); // Update the filter to process the data
        renderWindow->Render();


    }

    renderWindowInteractor->Start();
    return 0;

}
