/**
 * Project: Manus - Pure Integer Quantized Transformer
 * Author: Marcel (with Manus AI)
 * 
 * Description:
 * 100% Integer implementation of a Transformer block.
 * NO floats, NO math.h, NO floating-point unit required.
 * 
 * Scale Factor: 10^6 (1.0 = 1,000,000)
 */

#include <stdio.h>
#include <stdlib.h>

// --- Configuration ---
#define L 3         // Sequence Length
#define D_MODEL 4   // Model Dimension
#define D_FF 8      // Internal Feed-Forward Dimension

// Fixed-Point Scale Factor: 10^6
#define S 1000000LL

// --- Pure Integer Utilities ---

/**
 * Prints a fixed-point integer as a decimal string without using floats.
 * Example: 1500000 -> "1.500000"
 * FIXED: Correct handling of negative numbers
 */
void print_as_decimal(long long val) {
    int is_negative = (val < 0);
    if (is_negative) {
        printf("-");
        val = -val;
    }
    printf("%lld.%06lld", val / S, val % S);
}

void print_integer_matrix(const char *name, long long *mat, int rows, int cols) {
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        printf("  [");
        for (int j = 0; j < cols; j++) {
            print_as_decimal(mat[i * cols + j]);
            if (j < cols - 1) printf(", ");
        }
        printf("]\n");
    }
}

/**
 * Pure Integer Matrix Multiplication: C = (A * B) / S
 * FIXED: Use __int128 or split multiplication to avoid overflow
 */
void integer_matrix_multiply(long long *A, long long *B, long long *C, int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            __int128 sum = 0;  // Use 128-bit to prevent overflow
            for (int k = 0; k < colsA; k++) {
                sum += (__int128)A[i * colsA + k] * B[k * colsB + j];
            }
            C[i * colsB + j] = (long long)(sum / S);
        }
    }
}

void integer_matrix_transpose(long long *A, long long *B, int rowsA, int colsA) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsA; j++) {
            B[j * rowsA + i] = A[i * colsA + j];
        }
    }
}

/**
 * Linear Softmax Approximation (Integer only)
 */
void integer_softmax(long long *mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        long long sum = 0;
        for (int j = 0; j < cols; j++) sum += mat[i * cols + j];
        if (sum == 0) continue;
        for (int j = 0; j < cols; j++) {
            mat[i * cols + j] = ((__int128)mat[i * cols + j] * S) / sum;
        }
    }
}

// --- Transformer Components ---

void integer_attention(long long *Q, long long *K, long long *V, long long *Output) {
    long long *K_T = (long long *)malloc(D_MODEL * L * sizeof(long long));
    integer_matrix_transpose(K, K_T, L, D_MODEL);

    long long *Scores = (long long *)malloc(L * L * sizeof(long long));
    integer_matrix_multiply(Q, K_T, Scores, L, D_MODEL, L);

    // FIXED: Correct scaling by sqrt(D_MODEL) = sqrt(4) = 2
    // So divide by 2^2 = 4, but we already divided by S in multiply, so divide by 2 more
    for (int i = 0; i < L * L; i++) Scores[i] /= 2;

    integer_softmax(Scores, L, L);
    integer_matrix_multiply(Scores, V, Output, L, L, D_MODEL);

    free(K_T); free(Scores);
}

long long integer_relu(long long x) { return (x > 0) ? x : 0; }

void integer_ffn(long long *Input, long long *Output, long long *W1, long long *B1, long long *W2, long long *B2) {
    long long *Hidden = (long long *)malloc(L * D_FF * sizeof(long long));
    integer_matrix_multiply(Input, W1, Hidden, L, D_MODEL, D_FF);

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < D_FF; j++) {
            Hidden[i * D_FF + j] += B1[j];
            Hidden[i * D_FF + j] = integer_relu(Hidden[i * D_FF + j]);
        }
    }

    integer_matrix_multiply(Hidden, W2, Output, L, D_FF, D_MODEL);

    for (int i = 0; i < L; i++) {
        for (int j = 0; j < D_MODEL; j++) Output[i * D_MODEL + j] += B2[j];
    }
    free(Hidden);
}

void integer_residual(long long *Input, long long *Sub, long long *Output) {
    for (int i = 0; i < L * D_MODEL; i++) Output[i] = Input[i] + Sub[i];
}

// --- Main Block ---

void integer_transformer_block(long long *Input, long long *Output,
                               long long *W1, long long *B1, long long *W2, long long *B2) {
    long long *Att_Out = (long long *)malloc(L * D_MODEL * sizeof(long long));
    integer_attention(Input, Input, Input, Att_Out);

    long long *Res1 = (long long *)malloc(L * D_MODEL * sizeof(long long));
    integer_residual(Input, Att_Out, Res1);

    long long *FFN_Out = (long long *)malloc(L * D_MODEL * sizeof(long long));
    integer_ffn(Res1, FFN_Out, W1, B1, W2, B2);

    integer_residual(Res1, FFN_Out, Output);

    free(Att_Out); free(Res1); free(FFN_Out);
}

int main() {
    // Pure Integer Input (0.5, 0.1, 0.9, 0.2 ...)
    long long Input[L * D_MODEL] = {
        500000LL, 100000LL, 900000LL, 200000LL,
        100000LL, 800000LL, 200000LL, 700000LL,
        900000LL, 900000LL, 900000LL, 900000LL
    };

    // Pure Integer Weights
    long long W1[D_MODEL * D_FF] = {
        100000LL, 200000LL, 300000LL, 400000LL, 500000LL, 600000LL, 700000LL, 800000LL,
        800000LL, 700000LL, 600000LL, 500000LL, 400000LL, 300000LL, 200000LL, 100000LL,
        100000LL, 100000LL, 100000LL, 100000LL, 100000LL, 100000LL, 100000LL, 100000LL,
        900000LL, 900000LL, 900000LL, 900000LL, 900000LL, 900000LL, 900000LL, 900000LL
    };
    long long B1[D_FF] = {10000LL, 0LL, -10000LL, 20000LL, 0LL, 10000LL, -20000LL, 30000LL};

    long long W2[D_FF * D_MODEL] = {
        100000LL, 0LL, 0LL, 0LL, 0LL, 100000LL, 0LL, 0LL, 0LL, 0LL, 100000LL, 0LL, 0LL, 0LL, 0LL, 100000LL,
        100000LL, 100000LL, 100000LL, 100000LL, 0LL, 0LL, 0LL, 0LL, 200000LL, 200000LL, 200000LL, 200000LL, 0LL, 0LL, 0LL, 0LL
    };
    long long B2[D_MODEL] = {5000LL, -5000LL, 5000LL, -5000LL};

    long long Output[L * D_MODEL];

    printf("--- Manus: PURE INTEGER Transformer Block ---\n");
    print_integer_matrix("Input", Input, L, D_MODEL);

    integer_transformer_block(Input, Output, W1, B1, W2, B2);

    printf("\nFinal Transformed Output (Pure Integer Calculation):\n");
    print_integer_matrix("Output", Output, L, D_MODEL);

    return 0;
}
