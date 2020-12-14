///////////////////////////////////////////////////////////////////////
//
// CMPSC450 
//
// hw5_cbz5089.cpp
// 
// Author: Chenyin Zhang
//
// compile line:
// mpicc -o hw5_cbz5089.out hw5_cbz5089.cpp
// Notice: the program check the correctness on each processor after
// The calculation. If there is an error, the output log would include
// specify the rank of error processor
//
///////////////////////////////////////////////////////////////////////
#include <sys/time.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <unistd.h>

// complex algorithm for evaluation
void matrix_mult_orig(double *A, double *B, double *C, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            C[i * N + j] = 0;
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
}

void get_walltime(double* wcTime) {

     struct timeval tp;

     gettimeofday(&tp, NULL);

     *wcTime = (double)(tp.tv_sec + tp.tv_usec/1000000.0);

}

// serial matrix multiplication algorithm used in Cannon
void mm(int N, double *a, double *b, double *c) {
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            double temp = 0;
            for (int k = 0; k < N; k++) {
                temp += a[i * N + k] * b[k * N + j];
            }
            c[i * N + j] += temp;
        }
    }
}

void compareOutputs(double *output1, double *output2, int length, int rank) {
    for (int i = 0; i < length; i++)
        for (int j = 0; j < length; j++) {
            if (abs(output1[i * length + j] - output2[i * length + j]) > 0.000001) {
                printf("Error in Rank %d : Outputs do not match! (%i, %i) (%f, %f)\n", rank,
                       i, j,
                       output1[i * length + j], output2[i * length + j]);
                return;
            }
        }
}

/* Run with 100 processes, n-square */
void mm_MPI(MPI_Comm comm, double *a, double *b, double *c, int n) {
    int dim, coord[2];
    int source, dest;
    int left, right, up, down;
    int world_rank;
    int world_size;
    MPI_Status status;

    // Get the number of processes
    MPI_Comm_size(comm, &world_size);
    // Get the rank of the process
    MPI_Comm_rank(comm, &world_rank);
    // Get coords
    MPI_Cart_coords(comm, world_rank, 2, coord);

    // 4*4 grid
    dim = sqrt(world_size);
    // Variables inside each core
    int temp_n = n / dim;
    
	//MPI_Barrier(comm);

    // Create the buffer to handle the sendrecv() for the same address
    double *temp_buffer_a = new double[temp_n * temp_n];
	
	MPI_Request request;
	
    // Get the initial left shift for Matrix A on row nodes, notice tag is just an indication
    MPI_Cart_shift(comm, 1, -coord[0], &source, &dest);
    // DO the initial shift for submatrix A
    if (coord[0] != 0) {    // avoid interacting with the first row
        MPI_Isend(a, temp_n * temp_n, MPI_DOUBLE, dest, 1, comm, &request);
        MPI_Recv(temp_buffer_a, temp_n * temp_n, MPI_DOUBLE, source, 1, comm, &status);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        memcpy(a, temp_buffer_a, temp_n * temp_n * sizeof(double));
    }
	
    // Get the initial up shift for Matrix B on column nodes
    MPI_Cart_shift(comm, 0, -coord[1], &source, &dest);
    // DO the initial shift for submatrix B
    if (coord[1] != 0) {    // avoid interacting with the first col
		MPI_Isend(b, temp_n * temp_n, MPI_DOUBLE, dest, 1, comm, &request);
        MPI_Recv(temp_buffer_a, temp_n * temp_n, MPI_DOUBLE, source, 1, comm, &status);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        memcpy(b, temp_buffer_a, temp_n * temp_n * sizeof(double));
    }
    	
    // Get neighbour nodes
    MPI_Cart_shift(comm, 1, 1, &left, &right);
    MPI_Cart_shift(comm, 0, 1, &up, &down);
	
    for (int i = 0; i < dim; i++) {
		// compute the local result
        mm(temp_n, a, b, c);
        // Do the left shift for Matrix A by 1
        if (coord[1] == 0) {
            MPI_Recv(temp_buffer_a, temp_n * temp_n, MPI_DOUBLE, right, 0, comm, &status);
            MPI_Send(a, temp_n * temp_n, MPI_DOUBLE, left, 0, comm);
            memcpy(a, temp_buffer_a, temp_n * temp_n * sizeof(double));
        } else {
            MPI_Send(a, temp_n * temp_n, MPI_DOUBLE, left, 0, comm);
            MPI_Recv(a, temp_n * temp_n, MPI_DOUBLE, right, 0, comm, &status);
        }
        // Do the up shift for Matrix B by 1
        if (coord[0] == 0) {
            MPI_Recv(temp_buffer_a, temp_n * temp_n, MPI_DOUBLE, down, 0, comm, &status);
            MPI_Send(b, temp_n * temp_n, MPI_DOUBLE, up, 0, comm);
            memcpy(b, temp_buffer_a, temp_n * temp_n * sizeof(double));
        } else {
            MPI_Send(b, temp_n * temp_n, MPI_DOUBLE, up, 0, comm);
            MPI_Recv(b, temp_n * temp_n, MPI_DOUBLE, down, 0, comm, &status);
        }
    }
    free(temp_buffer_a);
}

int main(int argc, char **argv) {

    int N = 2000;
    
    if (argc >= 2)
	{ // 1st argument is N
		N = atoi(argv[1]);
	}
	
    double d_S, d_E;	// timer

	// Build up the MPI environment
    MPI_Init(&argc, &argv);
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
	if (rank == 0){
		printf("Working on the N = %d with %d of nodes\n", N, world_size);
	}
    const int NBLOCKS = sqrt(world_size);
    const int BLOCKSIZE = N / NBLOCKS;
    // declarations for buffer used in the program
    double *A = new double[N * N];
    double *B = new double[N * N];
    double *C = new double[N * N];
    double *orig_C = new double[N * N];

    double *block_a = new double[N * N / world_size];
    double *block_b = new double[N * N / world_size];
    double *block_c = new double[N * N / world_size];
    double *block_c_ans = new double[N * N / world_size];

    const int NPROWS = NBLOCKS;  /* number of rows in _decomposition_ */
    const int NPCOLS = NBLOCKS;  /* number of cols in _decomposition_ */
    const int BLOCKROWS = BLOCKSIZE;  /* number of rows in _block_ */
    const int BLOCKCOLS = BLOCKSIZE; /* number of cols in _block_ */

	// Populate the data and generate the serial MM results
    if (rank == 0) {
		for (int i = 0; i < N; i++)
		{
			for (int j = 0; j < N; j++)
			{
				A[i * N + j] = i * i;
				B[i * N + j] = (double)j / (double) (i + 1);
			}
		}
		get_walltime(&d_S);			// start benchmark
		
        matrix_mult_orig(A, B, orig_C, N);
        
		get_walltime(&d_E);        // end benchmark
        // report results
		printf("Elapsed time for serial code: %f\n", d_E - d_S);
    }
	
	// initialize the submatrix in each processor
    for (int ii = 0; ii < BLOCKROWS * BLOCKCOLS; ii++) {
        block_a[ii] = 0;
        block_b[ii] = 0;
        block_c[ii] = 0;
    }

	// Generate the 2-D mesh with MPI_Comm new_comm
    int dim[2], period[2], reorder;
    int coord[2], id;
    int source, dest;
    MPI_Comm new_comm;
    MPI_Status status;
    dim[0] = dim[1] = sqrt(world_size);
    period[0] = period[1] = true;
    reorder = true;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dim, period, reorder, &new_comm);

    // Get coords
    MPI_Cart_coords(new_comm, rank, 2, coord);
    // Get rank
    MPI_Comm_rank(new_comm, &rank);

    // Build up mpi block matrix type for passing submatrix
    MPI_Datatype blocktype;
    MPI_Datatype blocktype2;
    MPI_Type_vector(BLOCKROWS, BLOCKCOLS, N, MPI_DOUBLE, &blocktype2);
    MPI_Type_create_resized(blocktype2, 0, sizeof(double), &blocktype);
    MPI_Type_commit(&blocktype);

    int disps[NPROWS * NPCOLS];
    int counts[NPROWS * NPCOLS];
    for (int ii = 0; ii < NPROWS; ii++) {
        for (int jj = 0; jj < NPCOLS; jj++) {
            disps[ii * NPCOLS + jj] = ii * N * BLOCKROWS + jj * BLOCKCOLS;
            counts[ii * NPCOLS + jj] = 1;
        }
    }

    // Scatter the A B and answer_C blocks into each node
	MPI_Scatterv(A, counts, disps, blocktype, block_a, BLOCKROWS * BLOCKCOLS, MPI_DOUBLE, 0, new_comm);
    MPI_Scatterv(B, counts, disps, blocktype, block_b, BLOCKROWS * BLOCKCOLS, MPI_DOUBLE, 0, new_comm);
    MPI_Scatterv(orig_C, counts, disps, blocktype, block_c_ans, BLOCKROWS * BLOCKCOLS, MPI_DOUBLE, 0, new_comm);
	

    // Start comparing for the coordinates and rank
    for (int proc = 0; proc < world_size; proc++) {
        if (proc == rank) {
            // Check the rank vs coord
            if (rank / dim[0] != coord[0]) {
                printf("Rank check failed for coord[0] of Rank %d : %d != %d\n", rank, rank / dim[0], coord[0]);
            }
            if (rank % dim[0] != coord[1]) {
                printf("Rank check failed for coord[1] of Rank %d : %d != %d\n", rank, rank % dim[0], coord[1]);
            }
        }
        MPI_Barrier(new_comm);
    }
    // make sure all the processors are at the same position
	MPI_Barrier(new_comm);
	if(rank == 0){
		// start benchmark
		get_walltime(&d_S);
	}
	
    // Process Cannon's Algorithm
    mm_MPI(new_comm, block_a, block_b, block_c, N);
    
    // make sure all the processors are at the same position
    MPI_Barrier(new_comm);
    if(rank == 0){
		// end benchmark
		get_walltime(&d_E);
	}

    // Verify the result. If there is any issue, the processor would print the error
    for (int proc = 0; proc < world_size; proc++) {
        if (proc == rank) {
            compareOutputs(block_c_ans, block_c, BLOCKSIZE, rank);
        }
        MPI_Barrier(new_comm);
    }
    
	
	if(rank == 0){
		// report results
		printf("Elapsed time: %f\n", d_E - d_S);
	}
    MPI_Finalize();
	
	// free the buffers
	delete[] A;
	delete[] B;
	delete[] C;
	delete[] orig_C;
	
	delete[] block_a;
	delete[] block_b;
	delete[] block_c;
	delete[] block_c_ans;
	
    return 0;
}
