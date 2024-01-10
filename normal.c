#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define min(a, b) ((a) < (b) ? (a) : (b))

typedef struct {
    double x, y,z;
} coord;

coord get_vec(coord* a, coord* b) {
    return (coord){b->x - a->x, b->y - a->y, b->z - a->z};
}

double get_len(coord* a, coord* b, coord* c) {
    coord ba = get_vec(b, a);
    coord bc = get_vec(b, c);
    double ba_square_len = ba.x * ba.x + ba.y * ba.y + ba.z * ba.z;
    double bc_square_len = bc.x * bc.x + bc.y * bc.y + bc.z * bc.z;
    return sqrt(min(ba_square_len, bc_square_len));
}

struct halton_generator {
    int base;
    int n, d;
};

void init_halton_generator(struct halton_generator* gen, int base) {
    gen->base = base;
    gen->n = 0;
    gen->d = 1;
}

double next_halton(struct halton_generator* gen) {
    int x = gen->d - gen->n;
    if(x == 1) {
        gen->n = 1;
        gen->d *= gen->base;
    } else {
        int y = gen->d / gen->base;
        while(x <= y) {
            y /= gen->base;
        }
        gen->n = (gen->base + 1) * y - x; 
    }
    return gen->n / (double) gen->d;
}

struct coord_halton_generator {
    struct halton_generator g1, g2, g3;
};

struct base3 {
    int b1, b2, b3;
};

void init_coord_generator(struct coord_halton_generator* gen, struct base3 base) {
    init_halton_generator(&gen->g1, base.b1);
    init_halton_generator(&gen->g2, base.b2);
    init_halton_generator(&gen->g3, base.b3);
}

coord next_halton_coord(struct coord_halton_generator* gen) {
    return (coord){next_halton(&gen->g1), next_halton(&gen->g2), next_halton(&gen->g3)};
}

int main(int argc, char** argv) {
    if(argc < 4) {
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

    fscanf(input, "(%lf %lf %lf)\n(%lf %lf %lf)\n(%lf %lf %lf)",
        &a.x, &a.y, &a.z,
        &b.x, &b.y, &b.z,
        &c.x, &c.y, &c.z);
    fclose(input);
    
    FILE* output = fopen(argv[3], "w");
    if(output == NULL) { 
        puts("Output file couldn't be opened");
        exit(4);
    }
    double s = get_len(&a, &b, &c);

    double sq2 = sqrt(2);
    // analytic
    double v1 = s * s * s * sq2 / 3;

    // monte-carlo
    double d = s * sq2; // diameter

    struct base3 bases[16] = {
        {2, 3, 5},
        {3, 5, 7},
        {2, 5, 7},
        {4, 5, 9},
        {2, 11, 17},
        {2, 7, 11},
        {3, 4, 19},
        {4, 7, 9},
        {5, 6, 11},
        {11, 13, 17},
        {8, 9, 11},
        {3, 17, 19},
        {12, 17, 23},
        {14, 19, 25},
        {3, 17, 23},
        {5, 6, 13}
    };

    int cnt = 0;
    double begin = omp_get_wtime();

    if(thread_count == -1) {
        struct coord_halton_generator gen;
        init_coord_generator(&gen, bases[0]);
        for(int i = 0; i < n; i++) {
            coord c = next_halton_coord(&gen);
            if(c.x + c.y + c.z <= 1) {
                cnt++;
            }
        }
    } else {
        #pragma omp parallel num_threads(thread_count)
        {
            struct coord_halton_generator gen;
            init_coord_generator(&gen, bases[omp_get_thread_num()]);
            int local_cnt = 0;
            #pragma omp for schedule(static)
            for(int i = 0; i < n; i++) {
                coord c = next_halton_coord(&gen);
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
    printf("Time: %lf", omp_get_wtime() - begin);
    
    return 0;
}