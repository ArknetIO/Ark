#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    int64_t n = 20000000;
    if (argc > 1) n = atoll(argv[1]);

    const int64_t mod = 1000000007LL;
    int64_t acc = 0;

    for (int64_t i = 0; i < n; ++i) {
        acc = (acc + i) % mod;
    }

    printf("c checksum: %lld\n", (long long)acc);
    return 0;
}