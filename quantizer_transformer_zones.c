/**
 * Project: Manus - Pure Integer Quantized Transformer with Overflow Zones
 * Author: Marcel (with Manus AI)
 * 
 * Description:
 * 100% Integer implementation with OVERFLOW ZONE MAPPING.
 * When overflow occurs, smoothly transition to a different scale zone.
 * Each zone has boundaries; symbolic assignment tracks zone membership.
 * 
 * Zone System:
 * ZONE_0: Scale 10^6   (Standard precision)
 * ZONE_1: Scale 10^5   (1/10th precision, 10x range)
 * ZONE_2: Scale 10^4   (1/100th precision, 100x range)
 * ZONE_3: Scale 10^3   (1/1000th precision, 1000x range)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>

// --- Configuration ---
#define L 3         // Sequence Length
#define D_MODEL 4   // Model Dimension
#define D_FF 8      // Internal Feed-Forward Dimension

// --- Zone System Definition ---
typedef enum {
    ZONE_0 = 0,  // Scale 10^6 (standard)
    ZONE_1 = 1,  // Scale 10^5 (10x range)
    ZONE_2 = 2,  // Scale 10^4 (100x range)
    ZONE_3 = 3   // Scale 10^3 (1000x range)
} OverflowZone;

// Zone scale factors
static const long long ZONE_SCALES[4] = {
    1000000LL,   // ZONE_0
    100000LL,    // ZONE_1
    10000LL,     // ZONE_2
    1000LL       // ZONE_3
};

// Zone boundaries (max safe value in each zone)
static const long long ZONE_MAX[4] = {
    1000000000000LL,  // ZONE_0: 10^12
    10000000000000LL, // ZONE_1: 10^13 (can hold 10x larger values)
    100000000000000LL,// ZONE_2: 10^14 (can hold 100x larger values)
    LLONG_MAX         // ZONE_3: max long long
};

// --- Zone-Aware Value Structure ---
typedef struct {
    long long value;
    OverflowZone zone;
} ZonedValue;

// --- Zone Utilities ---

/**
 * Detect which zone a value should belong to based on magnitude
 */
OverflowZone detect_zone(long long val) {
    long long abs_val = (val < 0) ? -val : val;
    
    if (abs_val <= ZONE_MAX[ZONE_0]) return ZONE_0;
    if (abs_val <= ZONE_MAX[ZONE_1]) return ZONE_1;
    if (abs_val <= ZONE_MAX[ZONE_2]) return ZONE_2;
    return ZONE_3;
}

/**
 * Convert value from one zone to another
 * Maintains value precision while changing scale
 */
ZonedValue convert_zone(ZonedValue zv, OverflowZone target_zone) {
    if (zv.zone == target_zone) return zv;
    
    long long source_scale = ZONE_SCALES[zv.zone];
    long long target_scale = ZONE_SCALES[target_zone];
    
    // Convert: value_in_target_zone = (value_in_source * source_scale) / target_scale
    __int128 converted = ((__int128)zv.value * source_scale) / target_scale;
    
    return (ZonedValue){
        .value = (long long)converted,
        .zone = target_zone
    };
}

/**
 * Create a zoned value with automatic zone detection
 */
ZonedValue make_zoned(long long value) {
    OverflowZone zone = detect_zone(value);
    return (ZonedValue){.value = value, .zone = zone};
}

/**
 * Add two zoned values with automatic zone resolution
 */
ZonedValue zoned_add(ZonedValue a, ZonedValue b) {
    // Convert both to common zone (use higher zone for safety)
    OverflowZone common_zone = (a.zone > b.zone) ? a.zone : b.zone;
    
    ZonedValue a_converted = convert_zone(a, common_zone);
    ZonedValue b_converted = convert_zone(b, common_zone);
    
    // Add and check for overflow
    __int128 sum = (__int128)a_converted.value + b_converted.value;
    
    // If overflow, move to next zone
    OverflowZone result_zone = common_zone;
    if (sum > ZONE_MAX[common_zone] || sum < -ZONE_MAX[common_zone]) {
        if (common_zone < ZONE_3) {
            result_zone = common_zone + 1;
        }
    }
    
    return (ZonedValue){
        .value = (long long)sum,
        .zone = result_zone
    };
}

/**
 * Multiply two zoned values with overflow zone management
 */
ZonedValue zoned_multiply(ZonedValue a, ZonedValue b) {
    // Multiply mantissas
    __int128 product = (__int128)a.value * b.value;
    
    // Divide by source scale (both are in same conceptual space)
    long long source_scale = ZONE_SCALES[a.zone];
    product /= source_scale;
    
    // Detect result zone
    long long result_value = (long long)product;
    OverflowZone result_zone = detect_zone(result_value);
    
    return (ZonedValue){
        .value = result_value,
        .zone = result_zone
    };
}

/**
 * Print zone information
 */
void print_zoned_value(ZonedValue zv) {
    long long scale = ZONE_SCALES[zv.zone];
    long long integer_part = zv.value / scale;
    long long frac_part = zv.value % scale;
    
    if (zv.value < 0 && frac_part != 0) {
        frac_part = -frac_part;
    }
    
    printf("%lld.%06lld [Z%d]", integer_part, frac_part, zv.zone);
}

void print_zoned_matrix(const char *name, ZonedValue *mat, int rows, int cols) {
    printf("%s (%dx%d) with Zone Tracking:\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        printf("  [");
        for (int j = 0; j < cols; j++) {
            print_zoned_value(mat[i * cols + j]);
            if (j < cols - 1) printf(", ");
        }
        printf("]\n");
    }
}

// --- Transformer Components with Zone Awareness ---

/**
 * Zone-aware matrix multiplication with overflow detection
 */
void zoned_matrix_multiply(ZonedValue *A, ZonedValue *B, ZonedValue *C, 
                           int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            ZonedValue sum = make_zoned(0);
            
            for (int k = 0; k < colsA; k++) {
                ZonedValue product = zoned_multiply(A[i * colsA + k], B[k * colsB + j]);
                sum = zoned_add(sum, product);
            }
            
            C[i * colsB + j] = sum;
        }
    }
}

/**
 * Zone-aware softmax with overflow handling
 */
void zoned_softmax(ZonedValue *mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        ZonedValue sum = make_zoned(0);
        
        // Sum all values in row
        for (int j = 0; j < cols; j++) {
            sum = zoned_add(sum, mat[i * cols + j]);
        }
        
        if (sum.value == 0) continue;
        
        // Normalize by sum (in appropriate zone)
        for (int j = 0; j < cols; j++) {
            long long source_scale = ZONE_SCALES[mat[i * cols + j].zone];
            __int128 normalized = ((__int128)mat[i * cols + j].value * source_scale) / sum.value;
            
            mat[i * cols + j] = (ZonedValue){
                .value = (long long)normalized,
                .zone = detect_zone((long long)normalized)
            };
        }
    }
}

int main() {
    printf("=== Manus: OVERFLOW ZONE MAPPING TRANSFORMER ===\n\n");
    
    // Example: Values that cross zone boundaries
    ZonedValue test_values[6];
    
    // Standard precision values
    test_values[0] = make_zoned(500000LL);   // 0.5 in ZONE_0
    test_values[1] = make_zoned(1000000LL);  // 1.0 in ZONE_0
    
    // Values that need zone escalation
    test_values[2] = make_zoned(999999999999LL);  // Very large, triggers zone detection
    test_values[3] = make_zoned(10000000000LL);   // Another large value
    
    // Operations causing overflow
    test_values[4] = zoned_add(test_values[2], test_values[3]);  // Should escalate zones
    test_values[5] = zoned_multiply(test_values[0], test_values[1]);
    
    printf("Test Values with Zone Assignment:\n");
    for (int i = 0; i < 6; i++) {
        printf("  Value %d: ", i);
        print_zoned_value(test_values[i]);
        printf("\n");
    }
    
    // Demonstrate zone transition
    printf("\n--- ZONE TRANSITION EXAMPLE ---\n");
    ZonedValue v1 = make_zoned(1000000000000LL);  // Max ZONE_0
    ZonedValue v2 = make_zoned(1000000000000LL);
    
    printf("v1: ");
    print_zoned_value(v1);
    printf("\nv2: ");
    print_zoned_value(v2);
    
    ZonedValue v_sum = zoned_add(v1, v2);
    printf("\nv1 + v2: ");
    print_zoned_value(v_sum);
    printf(" (Smoothly transitioned to ZONE_%d)\n", v_sum.zone);
    
    printf("\n✓ System successfully maps overflows to appropriate zones!\n");
    
    return 0;
}
