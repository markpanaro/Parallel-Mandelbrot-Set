#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <mpi.h>

return void matToImage(char* filename, int* mat, int* dims);

int main(int argc, char** argv)
{
    int rank, numranks;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double starttime;
    double endtime;

    // resolution
    int nx = 600; // cols
    int ny = 400; // rows

    int maxiter = 255;

    int *mat;

    // imaginary numbers to use in Z=Z^2+C
    double cx; // real part of C
    double cy; // imag part of C
    double x;  // real part of Z
    double y;  // imag part of Z

    //"window" range
    double r_start = -2;
    double r_end = 1;
    double i_start = -1;
    double i_end = 1;

    int iter;

    // for computing the area of the Mandelbrot set
    double area;
    int numoutside = 0;

    mat = (int*)malloc(nx * ny * sizeof(int));

    int amount = ny / numranks;
    int place = rank * amount;
    int endpoint = place + amount;
    if (rank == numranks - 1)
    {
        place = place + (amount % numranks);
        amount = amount + (amount % numranks);
    }

    int* sendcounts = (int*)malloc(numranks * sizeof(int));
    int* displs = (int*)malloc(numranks * sizeof(int));

    int temp = (nx * ny) / numranks;

    // sendcounts[rank]=temp
    for (int i = 0; i < numranks; i++)
        sendcounts[i] = temp;

    // displs[rank]=temp*rank
    for (int i = 0; i < numranks; i++)
        displs[i] = temp * i;

    // pick a pixel

    starttime = MPI_Wtime();

// omp_set_num_threads(6)
#pragma omp parallel for reduction(+:x, y, cx, cy, iter, numoutside)
    for (int i = place; i < endpoint; i++)
    { // rows
        for (int j = 0; j < nx; j++)
        { // cols
            // convert pixel (i,j) to x and y for C=x+iy
            cx = r_start + 1.0 * j / nx * (r_end - r_start);
            cy = i_start + 1.0 * i / ny * (i_end - i_start);

            // Z_0=0 -> Z=x+iy
            x = 0;
            y = 0;
            iter = 0;

            // now loop (iterate over Z=Z^2+C)
            while (iter < maxiter)
            {
                iter++;
                double oldx = x;
                oldx = x;
                x = x * x - y * y + cx;
                y = 2 * oldx * y + cy;

                // if |z|^2 is beyond thresh
                if (x * x + y * y > 4)
                {
                    numoutside++;
                    break;
                }
            }

            // save to matrix to save image
            int locate = i - place; // adjust so that all elements will be at the start
            mat[locate * nx + j] = iter;
        }
    }

    endtime = MPI_Wtime();
    double commtime = endtime - starttime;

    MPI_Allreduce(MPI_IN_PLACE, &numoutside, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // area=total area of the region (2*3=6)*ratio of the number of pixels inside/total number of pixels
    area = (r_end - r_start) * (i_end - i_start) * (1.0 * nx * ny - numoutside) / (nx * ny);

    int* result = (int*)malloc(nx * ny * sizeof(int));

    int imagesize = nx * ny / numranks;

    MPI_Gatherv(mat, temp, MPI_INT, result, sendcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    printf("Area of Mandelbrot Set: %.5f\n", area);
    printf("Rank: %d Time: %.5f\n", rank, commtime);

    MPI_Finalize();

    int dims[2] = {ny, nx};

    if (rank == 0) // only rank 0 has the complete result array
        matToImage("image.jpg,result,dims");

    return (0);
}
