#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <omp.h>

// compile (f.e. as C++): clang++ -std=c++20 -D _CRT_SECURE_NO_WARNINGS -D _USE_MATH_DEFINES -O2 -fopenmp example.cc -o omp_example.exe
// run: .\omp_example.exe e_input e_output

int main(int argc, char *argv[])
{
    FILE *fin = fopen(argv[1], "rb");
    int n;
    fscanf(fin, "%i", &n);
    fclose(fin);  
    
    float tstart = omp_get_wtime();
    int sum = 0;
    #pragma omp parallel for
    for (int i = 0; i < n; ++i)
    {
      #pragma omp atomic
      sum += i;
    }
    float tend = omp_get_wtime();
    
    FILE *fout = fopen(argv[2], "wb");
    fprintf(fout, "%i\n", sum);
    fclose(fout);

    printf("Time (sec): %f", tend - tstart);
    
    return 0;
}
