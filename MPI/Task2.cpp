#include <iostream>
#include <mpi.h>
#include <chrono>

using namespace std;

MPI_Status status;

int main(int argc, char* argv[]) {
    int size = 4096;

    __int64* Am = new __int64[size * size];
    __int64* Bm = new __int64[size * size];
    __int64* Rs = new __int64[size * size];
    __int64* Rt = new __int64[size * size];

    for (int i = 0; i < size * size; i++) {
        Am[i] = rand() % 10;
        Bm[i] = rand() % 10;
        Rs[i] = 0;
        Rt[i] = 0;
    }

    auto startScalar = chrono::system_clock::now();

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            for (int k = 0; k < size; ++k) {
                Rs[i * size + j] += Am[i * size + k] * Bm[k * size + j];
            }
        }
    }

    auto endScalar = chrono::system_clock::now();
    chrono::duration<double> diffScalar = endScalar - startScalar;

    MPI_Init(&argc, &argv);

    int rank, sizeProc, source, blockSize, offset;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &sizeProc);

    if (rank == 0) {
        auto start = chrono::system_clock::now();
        blockSize = size / sizeProc;
        offset = 0;

        for (int dest = 1; dest < sizeProc; dest++) {
            int currentBlockSize = blockSize + (dest <= (size % sizeProc) ? 1 : 0);
            MPI_Send(&offset, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&currentBlockSize, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
            MPI_Send(&Am[offset * size], currentBlockSize * size, MPI_LONG_LONG, dest, 1, MPI_COMM_WORLD);
            MPI_Send(Bm, size * size, MPI_LONG_LONG, dest, 1, MPI_COMM_WORLD);
            offset += currentBlockSize;
        }

        for (int i = offset; i < offset + blockSize; i++) {
            for (int j = 0; j < size; j++) {
                Rt[i * size + j] = 0;
                for (int k = 0; k < size; k++) {
                    Rt[i * size + j] += Am[i * size + k] * Bm[k * size + j];
                }
            }
        }

        for (int i = 1; i < sizeProc; i++) {
            int currentBlockSize = blockSize + (i <= (size % sizeProc) ? 1 : 0);
            MPI_Recv(&offset, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&currentBlockSize, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
            MPI_Recv(&Rt[offset * size], currentBlockSize * size, MPI_LONG_LONG, i, 2, MPI_COMM_WORLD, &status);
        }

        auto end = chrono::system_clock::now();
        chrono::duration<double> diff = end - start;
        cout << "Time MPI: " << diff.count() << " s" << endl;
        cout << "Time scalar: " << diffScalar.count() << " s" << endl;

        bool equal = true;
        for (int i = 0; i < size * size; i++) {
            if (Rs[i] != Rt[i]) {
                equal = false;
                break;
            }
        }

        if (equal) {
            cout << "Matrix is equal" << endl;
        }
        else {
            cout << "Matrix is not equal" << endl;
        }

        delete[] Am;
        delete[] Bm;
        delete[] Rs;
        delete[] Rt;

    }
    else {
        MPI_Recv(&offset, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        MPI_Recv(&blockSize, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
        Am = new __int64[blockSize * size];
        MPI_Recv(Am, blockSize * size, MPI_LONG_LONG, 0, 1, MPI_COMM_WORLD, &status);
        Bm = new __int64[size * size];
        MPI_Recv(Bm, size * size, MPI_LONG_LONG, 0, 1, MPI_COMM_WORLD, &status);

        Rt = new __int64[blockSize * size];

        for (int i = 0; i < blockSize; i++) {
            for (int j = 0; j < size; j++) {
                Rt[i * size + j] = 0;
                for (int k = 0; k < size; k++) {
                    Rt[i * size + j] += Am[i * size + k] * Bm[k * size + j];
                }
            }
        }

        MPI_Send(&offset, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(&blockSize, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
        MPI_Send(Rt, blockSize * size, MPI_LONG_LONG, 0, 2, MPI_COMM_WORLD);

        delete[] Am;
        delete[] Bm;
        delete[] Rt;
    }

    MPI_Finalize();

    return 0;
}
