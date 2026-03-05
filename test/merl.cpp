#include <cstdio>
#include <cstdlib>
#include <cmath>

// Remove M_PI definition since it's already defined in math.h
#if !defined(_USE_MATH_DEFINES)
#define _USE_MATH_DEFINES
#endif

#define BRDF_SAMPLING_RES_THETA_H       90
#define BRDF_SAMPLING_RES_THETA_D       90
#define BRDF_SAMPLING_RES_PHI_D         360

#define RED_SCALE (1.0/1500.0)
#define GREEN_SCALE (1.15/1500.0)
#define BLUE_SCALE (1.66/1500.0)

// [All other functions remain exactly the same as before, just updating the includes]
void cross_product (double* v1, double* v2, double* out)
{
    out[0] = v1[1]*v2[2] - v1[2]*v2[1];
    out[1] = v1[2]*v2[0] - v1[0]*v2[2];
    out[2] = v1[0]*v2[1] - v1[1]*v2[0];
}

void normalize(double* v)
{
    double len = sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
    v[0] = v[0] / len;
    v[1] = v[1] / len;
    v[2] = v[2] / len;
}

void rotate_vector(double* vector, double* axis, double angle, double* out)
{
    double temp;
    double cross[3];
    double cos_ang = cos(angle);
    double sin_ang = sin(angle);

    out[0] = vector[0] * cos_ang;
    out[1] = vector[1] * cos_ang;
    out[2] = vector[2] * cos_ang;

    temp = axis[0]*vector[0]+axis[1]*vector[1]+axis[2]*vector[2];
    temp = temp*(1.0-cos_ang);

    out[0] += axis[0] * temp;
    out[1] += axis[1] * temp;
    out[2] += axis[2] * temp;

    cross_product(axis,vector,cross);
    
    out[0] += cross[0] * sin_ang;
    out[1] += cross[1] * sin_ang;
    out[2] += cross[2] * sin_ang;
}

void std_coords_to_half_diff_coords(double theta_in, double fi_in, double theta_out, double fi_out,
                                double& theta_half, double& fi_half, double& theta_diff, double& fi_diff)
{
    // compute in vector
    double in_vec_z = cos(theta_in);
    double proj_in_vec = sin(theta_in);
    double in_vec_x = proj_in_vec*cos(fi_in);
    double in_vec_y = proj_in_vec*sin(fi_in);
    double in[3]= {in_vec_x,in_vec_y,in_vec_z};
    normalize(in);

    // compute out vector
    double out_vec_z = cos(theta_out);
    double proj_out_vec = sin(theta_out);
    double out_vec_x = proj_out_vec*cos(fi_out);
    double out_vec_y = proj_out_vec*sin(fi_out);
    double out[3]= {out_vec_x,out_vec_y,out_vec_z};
    normalize(out);

    // compute halfway vector
    double half_x = (in_vec_x + out_vec_x)/2.0f;
    double half_y = (in_vec_y + out_vec_y)/2.0f;
    double half_z = (in_vec_z + out_vec_z)/2.0f;
    double half[3] = {half_x,half_y,half_z};
    normalize(half);

    // compute theta_half, fi_half
    theta_half = acos(half[2]);
    fi_half = atan2(half[1], half[0]);

    double bi_normal[3] = {0.0, 1.0, 0.0};
    double normal[3] = {0.0, 0.0, 1.0};
    double temp[3];
    double diff[3];

    // compute diff vector
    rotate_vector(in, normal, -fi_half, temp);
    rotate_vector(temp, bi_normal, -theta_half, diff);
    
    // compute theta_diff, fi_diff    
    theta_diff = acos(diff[2]);
    fi_diff = atan2(diff[1], diff[0]);
}

inline int theta_half_index(double theta_half)
{
    if (theta_half <= 0.0)
        return 0;
    double theta_half_deg = ((theta_half / (M_PI/2.0))*BRDF_SAMPLING_RES_THETA_H);
    double temp = theta_half_deg*BRDF_SAMPLING_RES_THETA_H;
    temp = sqrt(temp);
    int ret_val = (int)temp;
    if (ret_val < 0) ret_val = 0;
    if (ret_val >= BRDF_SAMPLING_RES_THETA_H)
        ret_val = BRDF_SAMPLING_RES_THETA_H-1;
    return ret_val;
}

inline int theta_diff_index(double theta_diff)
{
    int tmp = int(theta_diff / (M_PI * 0.5) * BRDF_SAMPLING_RES_THETA_D);
    if (tmp < 0)
        return 0;
    else if (tmp < BRDF_SAMPLING_RES_THETA_D - 1)
        return tmp;
    else
        return BRDF_SAMPLING_RES_THETA_D - 1;
}

inline int phi_diff_index(double phi_diff)
{
    if (phi_diff < 0.0)
        phi_diff += M_PI;

    int tmp = int(phi_diff / M_PI * BRDF_SAMPLING_RES_PHI_D / 2);
    if (tmp < 0)    
        return 0;
    else if (tmp < BRDF_SAMPLING_RES_PHI_D / 2 - 1)
        return tmp;
    else
        return BRDF_SAMPLING_RES_PHI_D / 2 - 1;
}

void lookup_brdf_val(double* brdf, double theta_in, double fi_in,
              double theta_out, double fi_out, 
              double& red_val, double& green_val, double& blue_val,
              int& index)
{
    double theta_half, fi_half, theta_diff, fi_diff;
    
    std_coords_to_half_diff_coords(theta_in, fi_in, theta_out, fi_out,
               theta_half, fi_half, theta_diff, fi_diff);

    // Store indices for output
    int th_idx = theta_half_index(theta_half);
    int td_idx = theta_diff_index(theta_diff);
    int pd_idx = phi_diff_index(fi_diff);

    index = pd_idx +
           td_idx * BRDF_SAMPLING_RES_PHI_D / 2 +
           th_idx * BRDF_SAMPLING_RES_PHI_D / 2 * BRDF_SAMPLING_RES_THETA_D;

    red_val = brdf[index] * RED_SCALE;
    green_val = brdf[index + BRDF_SAMPLING_RES_THETA_H*BRDF_SAMPLING_RES_THETA_D*BRDF_SAMPLING_RES_PHI_D/2] * GREEN_SCALE;
    blue_val = brdf[index + BRDF_SAMPLING_RES_THETA_H*BRDF_SAMPLING_RES_THETA_D*BRDF_SAMPLING_RES_PHI_D] * BLUE_SCALE;

    if (red_val < 0.0 || green_val < 0.0 || blue_val < 0.0)
        fprintf(stderr, "Below horizon.\n");
}

bool read_brdf(const char *filename, double* &brdf)
{
    FILE *f = fopen(filename, "rb");
    if (!f)
        return false;

    int dims[3];
    fread(dims, sizeof(int), 3, f);
    int n = dims[0] * dims[1] * dims[2];
    if (n != BRDF_SAMPLING_RES_THETA_H *
         BRDF_SAMPLING_RES_THETA_D *
         BRDF_SAMPLING_RES_PHI_D / 2) 
    {
        fprintf(stderr, "Dimensions don't match\n");
        fclose(f);
        return false;
    }

    brdf = (double*) malloc (sizeof(double)*3*n);
    fread(brdf, sizeof(double), 3*n, f);

    fclose(f);
    return true;
}

int main(int argc, char *argv[])
{
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <brdf_file> <theta_in> <phi_in> <theta_out> <phi_out>\n", argv[0]);
        fprintf(stderr, "All angles should be in degrees\n");
        return 1;
    }

    const char *filename = argv[1];
    double* brdf;

    // Convert input angles from degrees to radians
    double theta_in = atof(argv[2]) * M_PI / 180.0;
    double phi_in = atof(argv[3]) * M_PI / 180.0;
    double theta_out = atof(argv[4]) * M_PI / 180.0;
    double phi_out = atof(argv[5]) * M_PI / 180.0;

    // Read BRDF data
    if (!read_brdf(filename, brdf)) {
        fprintf(stderr, "Error reading %s\n", filename);
        return 1;
    }

    // Calculate BRDF values and get indices
    double red, green, blue;
    int brdf_index;
    double theta_half, fi_half, theta_diff, fi_diff;

    // Calculate Rusinkiewicz coordinates
    std_coords_to_half_diff_coords(theta_in, phi_in, theta_out, phi_out,
                                 theta_half, fi_half, theta_diff, fi_diff);

    // Look up BRDF values
    lookup_brdf_val(brdf, theta_in, phi_in, theta_out, phi_out, red, green, blue, brdf_index);

    // Print results
    printf("\nInput angles (degrees):\n");
    printf("theta_in: %.2f, phi_in: %.2f\n", theta_in * 180.0 / M_PI, phi_in * 180.0 / M_PI);
    printf("theta_out: %.2f, phi_out: %.2f\n", theta_out * 180.0 / M_PI, phi_out * 180.0 / M_PI);

    printf("\nRusinkiewicz coordinates (degrees):\n");
    printf("theta_half: %.2f\n", theta_half * 180.0 / M_PI);
    printf("phi_half: %.2f\n", fi_half * 180.0 / M_PI);
    printf("theta_diff: %.2f\n", theta_diff * 180.0 / M_PI);
    printf("phi_diff: %.2f\n", fi_diff * 180.0 / M_PI);

    printf("\nBRDF indices:\n");
    printf("Base index: %d\n", brdf_index);
    printf("Red index: %d\n", brdf_index);
    printf("Green index: %d\n", brdf_index + BRDF_SAMPLING_RES_THETA_H*BRDF_SAMPLING_RES_THETA_D*BRDF_SAMPLING_RES_PHI_D/2);
    printf("Blue index: %d\n", brdf_index + BRDF_SAMPLING_RES_THETA_H*BRDF_SAMPLING_RES_THETA_D*BRDF_SAMPLING_RES_PHI_D);

    printf("\nBRDF values:\n");
    printf("Red: %f\n", red);
    printf("Green: %f\n", green);
    printf("Blue: %f\n", blue);

    free(brdf);
    return 0;
}