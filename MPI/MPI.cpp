#include <iostream>
#include <mpi.h>
#include <boost/uuid/detail/md5.hpp>
#include <boost/algorithm/hex.hpp>
#include <ctime>

std::string generateRandomPassword(int length) {
    static const char alphanum[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";

    std::string password;
    for (int i = 0; i < length; ++i) {
        password += alphanum[rand() % (sizeof(alphanum) - 1)];
    }
    return password;
}

std::string calculateMD5(const std::string& input) {
    boost::uuids::detail::md5 md5;
    md5.process_bytes(input.data(), input.size());

    boost::uuids::detail::md5::digest_type digest;
    md5.get_digest(digest);

    std::string md5Hash;
    boost::algorithm::hex(reinterpret_cast<const unsigned char*>(&digest), reinterpret_cast<const unsigned char*>(&digest) + sizeof(digest), std::back_inserter(md5Hash));
    return md5Hash;
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int passwordLength = 5;

    srand(time(NULL) + rank);

    std::string targetHash;
    bool passwordFound = false;

    // Генерация пароля и хэша в процессе с рангом 0
    if (rank == 0) {
        std::string generatedPassword = generateRandomPassword(passwordLength);
        std::cout << "Generated Password: " << generatedPassword << std::endl;

        targetHash = calculateMD5(generatedPassword);
        std::cout << "Generated Hash: " << targetHash << std::endl;

        // Отправка целевого хэша другим процессам
        for (int dest = 1; dest < size; ++dest) {
            MPI_Send(targetHash.c_str(), targetHash.size() + 1, MPI_CHAR, dest, 0, MPI_COMM_WORLD);
        }
    }
    else {
        // Принятие целевого хэша от процесса с рангом 0
        char receivedHash[33];
        MPI_Recv(receivedHash, 33, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        targetHash = receivedHash;
    }

    // Генерация паролей и сравнение хэшей в цикле
    while (!passwordFound) {
        std::string generatedPassword = generateRandomPassword(passwordLength);
        std::string generatedHash = calculateMD5(generatedPassword);

        if (generatedHash == targetHash) {
            std::cout << "Rank " << rank << ": Password found! Password: " << generatedPassword << ", Hash: " << generatedHash << std::endl;
            passwordFound = true; // Устанавливаем флаг, чтобы прекратить генерацию
        }

        // Каждый процесс проверяет флаг после каждой итерации цикла
        MPI_Allreduce(MPI_IN_PLACE, &passwordFound, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
