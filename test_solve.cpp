#include "mkl.h"
#include <iostream>


int main() {
    std::cout << "Testing a symmetric indefinite matrix" << std::endl;

    MKL_INT n = 40;
    MKL_INT nnz = 78;

    MKL_INT indices[78] = { 0,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,  8,  9,
        9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17,
       18, 18, 19, 19, 20, 20, 21, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26,
       26, 27, 27, 28, 28, 29, 29, 30, 30, 31, 31, 32, 32, 33, 33, 34, 34,
       35, 35, 36, 36, 37, 37, 38, 38, 39, 39};

    MKL_INT indptr[41] = { 0,  2,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
       33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65,
       67, 69, 71, 73, 75, 77, 78};

    float data[78] = {-1.,  1., -1.,  2., -1.,  2., -1.,  2., -1.,  2., -1.,  2., -1.,
        2., -1.,  2., -1.,  2., -1.,  2., -1.,  2., -1.,  2., -1.,  2.,
       -1.,  2., -1.,  2., -1.,  2., -1.,  2., -1.,  2., -1.,  2., -1.,
        2., -1.,  2., -1.,  2., -1.,  2., -1.,  2., -1.,  2., -1.,  2.,
       -1.,  2., -1.,  2., -1.,  2., -1.,  2., -1.,  2., -1.,  2., -1.,
        2., -1.,  2., -1.,  2., -1.,  2., -1.,  2., -1.,  2., -1.,  2.};

    float b[40] = {
        1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
        19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32., 33., 34.,
        35., 36., 37., 38., 39., 40.
    };

    float x[40];

    /*
    Matrix looks like:
    A = [[-1,  1,  0, ...,  0,  0,  0],
         [ 1,  0, -1, ...,  0,  0,  0],
         [ 0, -1,  2, ...,  0,  0,  0],
         ...,
         [ 0,  0,  0, ...,  2, -1,  0],
         [ 0,  0,  0, ..., -1,  2, -1],
         [ 0,  0,  0, ...,  0, -1,  2]]
    (except it's just the non-zero lower diagonal)

    It has an LDL^T decomposition of:
    L = [[ 1.,  0.,  0., ...,  0.,  0.,  0.],
         [ 1.,  1.,  0., ...,  0.,  0.,  0.],
         [ 0., -1.,  1., ...,  0.,  0.,  0.],
         ...,
         [ 0.,  0.,  0., ...,  1.,  0.,  0.],
         [ 0.,  0.,  0., ..., -1.,  1.,  0.],
         [ 0.,  0.,  0., ...,  0., -1.,  1.]]

    D = [[-1.,  0.,  0., ...,  0.,  0.,  0.],
         [ 0.,  1.,  0., ...,  0.,  0.,  0.],
         [ 0.,  0.,  1., ...,  0.,  0.,  0.],
         ...,
         [ 0.,  0.,  0., ...,  1.,  0.,  0.],
         [ 0.,  0.,  0., ...,  0.,  1.,  0.],
         [ 0.,  0.,  0., ...,  0.,  0.,  1.]]
    */

    MKL_INT mtype = -2; // real symmetric indefinite
    MKL_INT nrhs = 1;

    // Init handle
    _MKL_DSS_HANDLE_t handle[64];
    for(int i; i<64; ++i){
        handle[i] = NULL;
    }

    MKL_INT maxfct=1, mnum=1, msglvl=0;

    // init iparm array
    MKL_INT iparm[64];
    for(int i; i<64; ++i){
        iparm[i] = 0;
    }

    // set some defaults
    iparm[0] = 1;  // tell pardiso to not reset these values on the first call
    iparm[1] = 2;  // The nested dissection algorithm from the METIS
    iparm[3] = 0;  // The factorization is always computed as required by phase.
    iparm[4] = 2;  // fill perm with computed permutation vector
    iparm[5] = 0;  // The array x contains the solution; right-hand side vector b is kept unchanged.
    iparm[7] = 0;  // The solver automatically performs two steps of iterative refinement when perterbed pivots are obtained
    iparm[9] = 8;  // pivoting permutation
    iparm[10] = 0;  // No pivot scaling
    iparm[11] = 0;  // Solve a linear system AX = B (as opposed to A.T or A.H)
    iparm[12] = 0;  // No scaling
    iparm[17] = -1;  // Return the number of non-zeros in this value after first call
    iparm[18] = 0;  // do not report flop count
    iparm[20] = 1;  // Bunch-Kaufman pivoting
    iparm[23] = 0;  // classic (not parallel) factorization
    iparm[24] = 0;  // default behavoir of parallel solving
    iparm[26] = 1;  // Check the input matrix
    iparm[27] = 1;  // single precision
    iparm[33] = 0;  // optimal number of thread for CNR mode
    iparm[34] = 1;  // zero based indexing
    iparm[35] = 0;  // Do not compute schur complement
    iparm[36] = 0;  // use CSR storage format
    iparm[38] = 0;  // Do not use low rank update
    iparm[42] = 0;  // Do not compute the diagonal of the inverse
    iparm[55] = 0;  // Internal function used to work with pivot and calculation of diagonal arrays turned off.
    iparm[59] = 0;  // operate in-core mode

    MKL_INT perm[40];

    MKL_INT error;

    void *dummy;

    MKL_INT phase = 11;
    pardiso(handle, &maxfct, &mnum, &mtype, &phase,
             &n, data, indptr, indices, perm, &nrhs, iparm, &msglvl, &dummy, &dummy, &error);
    if ( error != 0 ){
        std::cout << "ERROR during symbolic factorization: "<< error << std::endl;
        return 1;
    }
    std::cout << "Reordering completed ... " << std::endl;
    std::cout << "Number of nonzeros in factors = " << iparm[17] << std::endl;
    std::cout << "Number of factorization MFLOPS = " << iparm[18] << std::endl;

    phase = 22;
    pardiso(handle, &maxfct, &mnum, &mtype, &phase,
             &n, data, indptr, indices, perm, &nrhs, iparm, &msglvl, &dummy, &dummy, &error);
    if ( error != 0 )
    {
        std::cout <<  "ERROR during numerical factorization: " << error << std::endl;
        return 2;
    }
    std::cout << "Factorization completed ... " << std::endl;

    phase = 33;
    pardiso(handle, &maxfct, &mnum, &mtype, &phase,
             &n, data, indptr, indices, perm, &nrhs, iparm, &msglvl, b, x, &error);
    if ( error != 0 ){
        std::cout << "ERROR during solution: " << error << std::endl;
        return 3;
    }
    std::cout << "Solve completed ... " << std::endl;
    std::cout << "The solution of the system is: " << std::endl;
    for (int i = 0; i < n; i++ ){
        std::cout << " x ["<<i<<"] = " <<  x[i] << std::endl;
    }

    return 0;
}