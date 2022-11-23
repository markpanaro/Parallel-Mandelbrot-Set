# Parallel implementation of the Mandelbrot Set

## I Introduction

The goal for this project was to parallelize a Mandelbrot Set implementation utilizing both MPI and OpenMP and generate a grayscale image of the result. The purpose of doing this exercise is to work with both MPI and OpenMP together. Combining these tools is ideal for supercomputer environments, as it allows for both easy node level parallelism alongside simple communication between nodes.

## II Design and Implementation

The serial Mandelbrot set implementation used was provided in a lecture for COMP3450 (Parallel and Distributed Computing). As such, the majority of the work done was focused on dividing up the loop iterations to properly balance the work for each MPI node as well as creating an OpenMP parallel region to further speed up the total calculation. When implementing, the outer for loop of the Mandelbrot code was edited to run across multiple MPI ranks. Int variables amount, place, and endpoint were created utilizing the rank variable, which allows for each rank to start work at the correct spot in the loop. Each rank gets its own horizontal strip of the image to generate. Additionally, the entire nested for loop block was turned into an Open MP parallel for region with a reduction clause. This also helped to increase the speed of the calculation. 

Here is the code snippet:

```C
int amount = ny/numranks;
int place = rank*numranks;
int endpoint = place+amount
if(rank==numranks-1){		
	place=place+(amount%numranks);
	amount=amount+(amount%numranks);
}
…
#pragma omp parallel for reduction(+:x,y,cx,cy,iter,numoutside)
	for(int i=place;i<endpoint;i++){
		for(int j=0;j<nx;j++){
			…
		}
	}
```
	
After the OpenMP parallel region and MPI work, the code uses MPI_Gatherv to consolidate the results on rank 0. Only the loops that are generating results are timed, as it is expected that communication will take a lengthier amount of time and has the potential to be inconsistent. The sendcounts and displs arrays are created earlier in the code. Each sendcounts rank is equivalent to nx*ny/numranks, or the full size of the image divided by the number of ranks which did calculations for the final result. The displs elements follow a similar pattern, instead multiplying the image size divided by numranks by the current rank to get the proper displacement. At the end of the code, final times for each rank and the total area of the set are outputted. The final product is stored in a new array result of the full image size, and matToImage is run to produce a Mandelbrot Set visual. 

## III Results

Below is the timing data for the code with the MPI and OpenMP implementations. It is evident from the results that there is a significant speedup taking place. By editing just the outer loop to create horizontal strips of work, the code’s execution time has been significantly decreased without the need for a massive overhaul or extensive additions. OpenMP helps as well in that it allows each rank to finish its own work more quickly. The workload of each rank should be distributed evenly based implementation. However, there were some outliers while testing, the cause of which remains undiagnosed. Regardless, this code successfully achieves parallelization on two fronts, ultimately leading to a rapid and improved completion time.

![This is an image](Images/Parallel%20Mandelbrot%20Image%201.jpg)
![This is an image](Images/Parallel%20Mandelbrot%20Image%202.png)
![This is an image](Images/Parallel%20Mandelbrot%20Image%203.png)
![This is an image](Images/Parallel%20Mandelbrot%20Image%204.png)