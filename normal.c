#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define min(a, b) ((a) < (b) ? (a) : (b))

typedef struct {
    float x, y,z;
} coord;

coord get_vec(coord* a, coord* b) {
    return (coord){b->x - a->x, b->y - a->y, b->z - a->z};
}

float get_len(coord* a, coord* b, coord* c) {
    coord ba = get_vec(b, a);
    coord bc = get_vec(b, c);
    float ba_square_len = ba.x * ba.x + ba.y * ba.y + ba.z * ba.z;
    float bc_square_len = bc.x * bc.x + bc.y * bc.y + bc.z * bc.z;
    return sqrtf(min(ba_square_len, bc_square_len));
}

struct xorshift64_generator {
    uint64_t a;
};

uint64_t xorshift64(struct xorshift64_generator *gen) {
	gen->a ^= gen->a << 13;
	gen->a ^= gen->a >> 7;
	gen->a ^= gen->a << 17;
	return gen->a;
}

void init_xorshift64_generator(struct xorshift64_generator* gen, uint64_t state) {
    gen->a = state;
}

float next_float(struct xorshift64_generator* gen) {
   return (uint32_t)xorshift64(gen) / (float)0xFFFFFFFFu;
}

struct coord_generator {
    struct xorshift64_generator g1, g2, g3;
};

struct triple_state {
    uint64_t s1, s2, s3;
};

void init_coord_generator(struct coord_generator* gen, struct triple_state *states) {
    init_xorshift64_generator(&gen->g1, states->s1);
    init_xorshift64_generator(&gen->g2, states->s2);
    init_xorshift64_generator(&gen->g3, states->s3);
}

coord next_coord(struct coord_generator* gen) {
    return (coord){next_float(&gen->g1), next_float(&gen->g2), next_float(&gen->g3)};
}

int main(int argc, char** argv) {

    if(argc != 4) {
        puts("Invalid number of arguments");
        exit(1);
    }
    int thread_count = atoi(argv[1]);
    if(thread_count < -1) {
        puts("Invalid number of threads");
        exit(2);
    }

    FILE* input = fopen(argv[2], "r");
    if(input == NULL) {
        puts("Input file couldn't be opened");
        exit(3);
    }

    int n;
    fscanf(input, "%d\n", &n); 
    coord a, b, c;

    fscanf(input, "(%f %f %f)\n(%f %f %f)\n(%f %f %f)",
        &a.x, &a.y, &a.z,
        &b.x, &b.y, &b.z,
        &c.x, &c.y, &c.z);
    fclose(input);
    
    FILE* output = fopen(argv[3], "w");
    if(output == NULL) { 
        puts("Output file couldn't be opened");
        exit(4);
    }

    float s = get_len(&a, &b, &c);

    float sq2 = sqrt(2);

    // analytic
    float v1 = s * s * s * sq2 / 3;

    // monte-carlo
    float d = s * sq2; // diameter

    struct triple_state bases[16] = { // some random seed states
        {0xc96d191cf6f6aea6,0x401f7ac78bc80f1c,0xb5ee8cb6abe457f8},
        {0xf258d22d4db91392,0x4eef2b4b5d860cc,0x67a7aabe10d172d6},
        {0x40565d50e72b4021,0x5d07b7d1e8de386,0x8548dea130821acc},
        {0x583c502c832e0a3a,0x4631aede2e67ffd1,0x8f9fccba4388a61f},
        {0x23d9a035f5e09570,0x8b3a26b7aa4bcecb,0x859c449a06e0302c},
        {0xdb696ab700feb090,0x7ff1366399d92b12,0x6b5bd57a3c9113ef},
        {0xbe892b0c53e40d3d,0x3fc97b87bed94159,0x3d413b8d11b4cce2},
        {0x51efc5d2498d7506,0xe916957641c27421,0x2a327e8f39fc19a6},
        {0x3edb3bfe2f0b6337,0x32c51436b7c00275,0xb744bed2696ed37e},
        {0xf7c35c861856282a,0xc4f978fb19ffb724,0x14a93ca1d9bcea61},
        {0x75bda2d6bffcfca4,0x41dbe94941a43d12,0xc6ec7495ac0e00fd},
        {0x957955653083196e,0xf346de027ca95d44,0x702751d1bb724213},
        {0x528184b1277f75fe,0x884bb2027e9ac7b0,0x41a0bc6dd5c28762},
        {0xba88011cd101288,0x814621bd927e0dac,0xb23cb1552b043b6e},
        {0x175a1fed9bbda880,0xe838ff59b1c9d964,0x7ea06b48fca72ac},
        {0x26ebdcf08553011a,0xfb44ea3c3a45cf1c,0x9ed34d63df99a685}
    };

    int cnt = 0;

    int available = omp_get_num_procs();

    double begin = omp_get_wtime();

    if(thread_count == -1) {
        struct coord_generator gen;
        init_coord_generator(&gen, &bases[0]);
        for(int i = 0; i < n; i++) {
            coord c = next_coord(&gen);
            if(c.x + c.y + c.z <= 1) {
                cnt++;
            }
        }
    } else {

        if(thread_count > available) {
            thread_count = available;
        }

        #pragma omp parallel num_threads(thread_count)
        {
            struct coord_generator gen;
            init_coord_generator(&gen, &bases[omp_get_thread_num()]);
            int local_cnt = 0;
            #pragma omp for schedule(static)
            for(int i = 0; i < n; i++) {
                coord c = next_coord(&gen);
                if(c.x + c.y + c.z <= 1) {
                    local_cnt++;
                }
            }
            #pragma omp atomic
            cnt += local_cnt;
        }
    }

    fprintf(output, "%g %g\n", v1, d * d * d * cnt / n);
    fclose(output);
    
    printf("Time (%i thread(s)): %g ms\n", 
        thread_count == -1 ? 0 : thread_count == 0 ? available : thread_count,
        1000 * (omp_get_wtime() - begin));
        
    return 0;
}